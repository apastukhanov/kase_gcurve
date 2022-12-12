from datetime import datetime, timedelta
from typing import List, Tuple

import logging

from pathlib import Path

import sqlite3

import pandas as pd
import numpy as np
import requests
from xlrd.sheet import Sheet

from config import (WAPR_URL_TEMPLATE, 
                    REQUEST_HEADERS, 
                    BASE_EXCEL_WAPR_PATH,
                    BASE_CSV_WAPR_PATH,
                    DB_WAPR_PATH)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def connect_to_kase(url: str, excel_path: Path) -> None:
    logger.info(f"Connecting to {url}")
    s = requests.Session()
    s.headers = REQUEST_HEADERS
    r = s.get(url)
    logger.info(f"{r.status_code=}")
    
    with open(excel_path, 'wb') as f:
        f.write(r.content)


def create_short_date(date: datetime) -> str:
    year = date.year - 2000
    td = date.strftime("%d%m") + str(year)
    return td
    

def download_wapr(tradedate: datetime) -> None:
    td = create_short_date(tradedate)
    url = WAPR_URL_TEMPLATE.format(td)
    connect_to_kase(url, Path(BASE_EXCEL_WAPR_PATH.format(td)))


def get_parsed_df(tradedate: datetime, is_gov: bool = False):
    td = tradedate
    file = Path(f"{BASE_CSV_WAPR_PATH}/{td.year}/"
                f"{td.strftime('%m')}/wapr{td.strftime('%Y%m%d')}.csv")
    if not file.exists():
        logger.info("downloading file from web..")
        df = read_excel_file(tradedate)
    else:
        logger.info("reading csv file from folder..")
        df = pd.read_csv(file)
    
    if is_gov:
        return df.loc[(df['Класс актива'].str.lower().str.contains('гос')) 
                      & (df['Средневзв. доходность, % годовых'].notna()) 
                      & (df['ISIN'].str.startswith('KZ'))]
    return df 


def read_excel_file(tradedate: datetime):
    td = create_short_date(tradedate)
    excel_path = Path(BASE_EXCEL_WAPR_PATH.format(td))
    if not excel_path.exists():
        download_wapr(tradedate)
    assert excel_path.parent.exists()
    all_df = parse_all_tables_from_xls_file(excel_path)
    df = pd.concat(all_df)
    df = df.replace('–', np.NAN).replace('-', np.NAN)
    df['tradedate'] = tradedate
    return df


def get_secids_on_date(tradedate: datetime, is_gov: bool = False) -> List[str]:
    df = get_parsed_df(tradedate)
    print(df.columns)
    if is_gov:
        return df.loc[(df['Класс актива'].str.lower().str.contains('гос')) 
                      & (df['Средневзв. доходность, % годовых'].notna()) 
                      & (df['ISIN'].str.startswith('KZ')), 'Торговый код'].values
    return df['Торговый код'].values 


def parse_all_tables_from_xls_file(filepath: str) -> List[pd.DataFrame]:
    all_df = []
    
    f = pd.ExcelFile(filepath)

    sheets = f.sheet_names
    sheets = [x for x in sheets if not "поясн" in x.lower()]

    for s in sheets:
        sh = f.book.sheet_by_name(s)
        start, end, lc = find_bouders_of_table_header(sh)

        assert start != -1 and end != -1 and lc != 0

        cols = parse_cols_on_sheet(sh, start, end, lc)
        df = pd.read_excel(f, sheet_name=s, 
                           skiprows=end, names=cols)
        col_to_del = list(filter(lambda x: x.strip() == '', 
                                 df.columns))
        df.drop(col_to_del, axis=1, inplace=True)
        df['Класс актива'] = s
        all_df.append(df)
        
    return all_df


def parse_cols_on_sheet(sheet: Sheet, start_row: int, 
                        end_row: int, last_col: int) -> List[str]:
    cols = []
    for j in range(min(last_col+1, sheet.ncols)):
        col_name = []
        for i in range(start_row, end_row+1):
            c = sheet.cell(i, j)
            if c.value != " ":
                col_name.append(c.value.strip())
        cols.append(" ".join(col_name))
    return cols


def find_bouders_of_table_header(sheet: Sheet) -> Tuple[int, int, int]:
    start = -1
    end = -1
    for i in range(min(20, sheet.nrows)):
        if str(sheet.cell(i, 0).value).lower() == 'торговый':
            start = i
        if sheet.cell(i, 0).value ==' ':
            end = i
    
    lc = 0
    while lc < sheet.ncols and \
            sheet.cell(start, lc).value != "":
        lc+=1
    
    return start, end, lc


def save_wapr_to_csv_db(df:pd.DataFrame, 
                    tradedate: datetime) -> None:
    td = tradedate
    file = Path(f"{BASE_CSV_WAPR_PATH}/{td.year}/"
                f"{td.strftime('%m')}/wapr{td.strftime('%Y%m%d')}.csv")
    file.parents[0].mkdir(parents=True, exist_ok=True)
    df.to_csv(file, index=False)
    

def save_wapr_to_sql(df: pd.DataFrame) -> None:
    with sqlite3.connect(DB_WAPR_PATH) as conn:
        df['updated_time'] = datetime.now()
        df.to_sql('WAPR', conn, if_exists='append', index=False)
    
     
def main(tradedate: datetime) -> None:
    df = read_excel_file(tradedate)
    logger.info(df.head())
    save_wapr_to_csv_db(df, tradedate)
    save_wapr_to_sql(df)
    

if __name__ == "__main__":
    d = datetime.now()
    bod = datetime(d.year, d.month, d.day-2)
    logger.info((bod-timedelta(3)).weekday())
    days_count = (bod.weekday() - 4) if bod.weekday() in [5, 6, 0] else 1
    days_count = 3 if days_count < 0 else days_count
    main(bod - timedelta(days=days_count))
    # main(datetime(2022,12,7))
    # print(get_secids_on_date(bod - timedelta(days=days_count), True))

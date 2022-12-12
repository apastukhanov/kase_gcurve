from datetime import datetime
import math
import os
from typing import List, Dict

from functools import partial

from bs4 import BeautifulSoup

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

import pandas as pd
import numpy as np

import requests

from config import HEADERS, FILE_ENCODING


def download_gcurve_params() -> None:
    URL = "https://kase.kz/ru/documents/curve"
    r = requests.get(URL)
    with open("output.xls", "wb") as f:
        f.write(r.content)


def get_tradedates() -> List:
    df = get_df_with_params()
    return df['tradedate'].sort_values(ascending=True)


def get_df_with_params() -> pd.DataFrame:
    df = pd.read_excel('output.xls')
    df = df.iloc[3:].reset_index(drop=True)

    df.columns = ['tradedate', 'B0', 'B1', 'B2', 'TAU']

    df['tradedate'] = pd.to_datetime(df['tradedate'], format='%Y-%m-%d %H:%M:%S')
    df[['B0', 'B1', 'B2', 'TAU']] = df[['B0', 'B1', 'B2', 'TAU']].astype('float64')
    return df


def get_gcurve_yield(tradedate: datetime, m: float) -> float:
    '''
    #print(datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S"))
    #download_gcurve_params()
    # m = 5.003
    # m = 5.336 #11.387
    # m = 5.392 #0.112805
    # print(get_gcurve_yield(datetime(2022,11,23), m))
    '''
    df = get_df_with_params()
    row = df.loc[df['tradedate'] == tradedate]
    if row.shape[0] < 1:
        return None
    return (row["B0"] + ((row["B1"] + row["B2"]) * (
            row["TAU"] / m) * (1 - math.exp(-m / row["TAU"])))
            - row["B2"] * math.exp(-m / row["TAU"])).values[0]


def plot_gcurve(tradedate: datetime, flag_renew_plot=True) -> None:
    dur = [0.25, 0.5, 0.75, 1, 2, 3, 5.392, 10, 15, 20, 25, 30]
    tradedates = [tradedate for i in range(len(dur))]
    values = list(map(get_gcurve_yield, tradedates, dur))

    plt_df = pd.DataFrame.from_dict({'duration': dur, 'yield': values})

    if flag_renew_plot:
        plt.clf()

    g = sns.lineplot(data=plt_df,
                     x='duration',
                     y='yield', legend='brief',
                     label=f"{tradedate.day}-{tradedate.month}-{tradedate.year}").set(
        title=datetime.strftime(tradedate, "G-curve as for %d.%m.%Y"))

    plt.legend(loc='best')

    plt.show()


def animate_gcurve():
    fig = plt.figure()

    anim = FuncAnimation(fig, partial(plot_gcurve, flag_renew_plot=False),
                         frames=get_tradedates().iloc[-10:],
                         interval=700, repeat=False)
    plt.show()


def parse_trades(tradedate, market='shares', price_filter=''):
    '''
    market:: gsecs, shares
    '''
    url = f'https://kase.kz/ru/trade_information/ajax/{market}/{price_filter}'
    s = requests.session()
    s.headers.update(HEADERS)
    r = s.post(url, data={'date': tradedate})
    with open('output.html', 'w', encoding='utf8') as f:
        print(r.text.encode('cp1250'))
        f.write(r.text)
    df = parse_trades_html(r.text)
    if df.shape[0] < 1:
        return None
    df = df.loc[~(df.iloc[:, 1].str.lower().str.contains('итого'))]
    df['tradedate'] = tradedate
    cols_float = ['Лучшая котировка на покупку', 'Лучшая котировка на продажу',
                  'Цена первой сделки', 'Максимальная цена', 'Минимальная цена',
                  'Цена последней сделки', 'Изменение цены за день, %',
                  'Средневзвеш. цена', 'Объем, млн KZT']
    for col in cols_float:
        df[col] = df[col].str.replace(',', '.').str.replace("–", "0").str.replace(" ", "").astype('float64')
    df['Количество сделок'] = df['Количество сделок'].str.replace("–", "0").astype('int64')
    df['Дни до погашения'] = df['Дни до погашения'].str.replace("–", "0").str.replace('бессрочные', '1000').astype(
        'int64')

    df = df.loc[df['Объем, млн KZT'] > 0]
    df["Yield, %"] = 0.0
    df["Duration, years"] = 0.0

    sht_cols = ['tradedate',
                'Код', 'Режим', 'Минимальная цена', 'Максимальная цена',
                'Цена первой сделки', 'Цена последней сделки', 'Изменение цены за день, %',
                'Количество сделок', 'Объем, млн KZT',
                'Средневзвеш. цена', 'Yield, %', 'Дни до погашения']

    df = df[sht_cols]
    df = df.sort_values('Дни до погашения')
    df.to_csv(f'trades/{market}_{price_filter}_{tradedate}.csv',
              index=False, sep=';', encoding=FILE_ENCODING)
    return df


def get_trades_from_file(tradedate: datetime):
    df = pd.read_csv('trades/all_trades.csv', encoding=FILE_ENCODING)
    df['tradedate'] = pd.to_datetime(df['tradedate'],
                                     format='%Y-%m-%d %H:%M:%S')
    df = df.loc[df['tradedate'] == tradedate]
    df["Yield, %"] = 0.0
    df["Duration, years"] = 0.0
    return df


def parse_sec_info(isin: str):
    file_path = f"sec_info/{isin}.csv"
    if not os.path.exists(file_path):
        url = f"https://kase.kz/ru/gsecs/show/{isin}/"
        s = requests.session()
        s.headers.update(HEADERS)
        r = s.get(url)
        with open(f'sec_info_html/sec_info_{isin}.html', 'w', encoding='utf8') as f:
            f.write(r.text)
        return parse_sec_info_html(r.text)
    return pd.read_csv(file_path)


def create_df_from_params() -> pd.DataFrame:
    params = get_df_with_params()
    dur = [0.25, 0.5, 0.75, 1, 2, 3, 5, 10, 15, 20, 25, 30]

    df = pd.DataFrame()

    def calc_g_curve(row, m):
        return (row["B0"] \
                + ((row["B1"] + row["B2"]) * (row["TAU"] / m) * (1 - math.exp(-m / row["TAU"]))) \
                - row["B2"] * math.exp(-m / row["TAU"]))

    for num, row in params.iterrows():
        tradedate = row['tradedate']
        rows = [row for i in range(len(dur))]
        values = list(map(calc_g_curve, rows, dur))
        tmp_dict = {'tradedate': tradedate}
        tmp_dict.update({k: v for k, v in zip(dur, values)})
        df = pd.concat([df, pd.DataFrame(tmp_dict, index=[0])], ignore_index=True)

    return df


def create_df_from_params_vect():
    params = get_df_with_params()
    # dur = [0.25, 0.5, 0.75, 1, 2, 3, 5, 10, 15, 20, 25, 30]
    dur = np.linspace(0.25, 30, 120)
    df = pd.DataFrame(columns=[f'{d}' for d in dur])

    def calc(row, m):
        return (row["B0"] \
                + ((row["B1"] + row["B2"]) * (row["TAU"] / m) * (1 - math.exp(-m / row["TAU"]))) \
                - row["B2"] * math.exp(-m / row["TAU"]))

    for d in dur:
        df[f'{d}'] = params.apply(partial(calc, m=d), axis=1)
    return pd.concat([params['tradedate'], df], axis=1)


def parse_trades_html(content: str):
    soup = BeautifulSoup(content, features='lxml')
    if len(soup.find_all(class_=
                 r'alert alert-danger')) > 0:
        print(soup.find(class_=
                        r'alert alert-danger').text)
        return None
    columns = [[col.text.strip() for col in row.find_all('th')]
               for row in soup.find('thead').find_all('tr')][0]
    data = [{columns[i]: col.text.strip()
             for i, col in enumerate(row.find_all('td'))}
            for row in soup.find('tbody').find_all('tr')
            if "Итого" not in row.find('td').text]
    return pd.DataFrame(data)
    # except Exception as e:
    #     print(content)


def parse_sec_description(soup: BeautifulSoup) -> Dict:
    try:
        data = [[k.text.strip() for k in row.find_all(class_='info-table__cell')]
                for row in soup.find(class_='info-table') \
                    .find_all(class_='info-table__row')]
    except Exception as e:
        return None
    data_dict = {row[0]: row[1] for row in data}
    return data_dict


def parse_sec_coupons(soup: BeautifulSoup) -> List[Dict]:
    try:
        coupons = soup.find_all(class_='modal-content')[0]
        # print(f"{coupons=}")
        # print(f"{coupons.find_all('tr')=}")
        columns = [[col.text.strip() for col in row.find_all('th')]
                   for row in coupons.find('thead').find_all('tr')][0]
    except Exception as e:
        print(str(e))
        return None

    data = [{columns[i]: col.text.strip()
             for i, col in enumerate(row.find_all('td'))}
            for row in coupons.find('tbody').find_all('tr')]

    if len(data) < 2:
        return None
    
    # print(data)
    
    return data


def get_tonia(td: datetime) -> float:
    url = f'https://kase.kz/ru/money_market/repo-indicators/tonia/archive-xls/{td.strftime("%d.%m.%Y")}/{td.strftime("%d.%m.%Y")}'
    df = pd.read_excel(url, skiprows=1)
    return df['Закрытие'].values[0]
    

def parse_sec_info_html(content: str):
    soup = BeautifulSoup(content, features='lxml')
    data_dict = parse_sec_description(soup)
    data = parse_sec_coupons(soup)
    
    if data:
        coupons_df = pd.DataFrame(data)
        coupons_df["Дата начала купонной выплаты"] = \
            pd.to_datetime(coupons_df["Дата начала купонной выплаты"].apply(
                lambda x: f"{x.split('.')[0]}.{x.split('.')[1]}.20{x.split('.')[2]}"),
                format="%d.%m.%Y")
        freq = coupons_df["Дата начала купонной выплаты"].dt.year.value_counts().max()
        coupons_df['freq'] = freq
        coupons_df['Ставка, % год.'] = coupons_df['Ставка, % год.'] \
                                                .str.replace(',', '.') \
                                                .str.replace('–', '0').astype('float64')
    
    else: 
        coupons_df = pd.DataFrame()
        
    if data_dict:
        
        if coupons_df.shape[0] < 1:
            cleaned_dict = {k.replace(':',''): v for k,v in data_dict.items()}
            coupons_df = pd.DataFrame([cleaned_dict])
        else:
            coupons_df["Список ценных бумаг"] = data_dict["Список ценных бумаг:"]
            coupons_df["Валюта котирования"] = data_dict["Валюта котирования:"]
            coupons_df["ISIN"] = data_dict["ISIN:"]

            if "Дата погашения:" in data_dict.keys():
                coupons_df["Дата погашения"] = data_dict["Дата погашения:"]

    if coupons_df.shape[0] > 1:
        # print(coupons_df)
        coupons_df.to_csv(f'sec_info/{data_dict["Код бумаги:"]}.csv',
                        index=False,
                        date_format="%d.%m.%Y")

        return coupons_df
    return None


if __name__ == '__main__':
    # download_gcurve_params()
    # # animate_gcurve()
    # parse_trades('08.12.2022', 'gsecs', '#gsec_clean')
    print(parse_sec_info('NTK028_2816'))
    # plot_gcurve(datetime(2022, 11, 16))
    # plot_gcurve_last()
    # parse_sec_info()
    # tic = time.perf_counter()
    # print(create_df_from_params().head())
    # toc = time.perf_counter()
    # print(f'func speed is {toc-tic:0.4f} seconds')
    # func speed is 148.2893 seconds
    # func speed is 148.0731 seconds
    # func speed is 148.3251 seconds
    # tic = time.perf_counter()
    # print(create_df_from_params_vect().head())
    # toc = time.perf_counter()
    # print(f'func speesd is {toc-tic:0.4f} seconds')
    # parse_trades_html()
    # print(get_trades_from_file(datetime(2022,12,1,21,0,0)))
    # plot_gcurve(datetime(2022, 12, 1))
    # print(FILE_ENCODING)
    get_tonia(datetime(2022, 12, 9))

from datetime import datetime
import json
import os

from typing import Dict, List
from pathlib import Path

import pandas as pd
import numpy as np

import PublicApiClient as NtApi


pub_ = os.getenv("PUBLIC_KEY")
sec_ = os.getenv("SECRET_KEY")


def write_to_json(content, filename)->None:
    filename = filename if filename else "response"
    with open(f'output/{filename}.json', 'w', encoding='utf8') as f:
        json.dump(content, f, indent=2, ensure_ascii=False)


def mk_request(cmd_:str, params_:Dict={}, isV1=False, filename=None,
               url:str = None)->None:
    res = NtApi.PublicApiClient(pub_, sec_, NtApi.PublicApiClient().V2)
    if isV1:
        res = NtApi.PublicApiClient(pub_, sec_, NtApi.PublicApiClient().V1)
    if url:
        res.setApiUrl(url)
    j = res.sendRequest(cmd_, params_).json()
    write_to_json(j, filename=filename)
    return j


def search_ticker(search_str: str)->None:
    cmd_ = "tickerFinder"
    params_ = {
        "text": search_str
    }
    return mk_request(cmd_, params_, True, filename=f'find_res/{search_str}.json')
     

def get_user_data()->None:
    cmd_ = "getOPQ"
    mk_request(cmd_)
    

def get_trade_hist(ticker, from_, to_, timeframe=1440):
    cmd_   = 'getHloc'
    params_ = {
        'id'           : ticker,
        'count'        : -1,
        'timeframe'    : timeframe,
        # "date_from"    : '16.08.2020 00:00',
        # "date_to"      : '17.08.2020 00:00',
        'date_from'    : datetime.strftime(from_, "%d.%m.%Y %H:%M"),
        'date_to'      : datetime.strftime(to_, "%d.%m.%Y %H:%M"),
        'intervalMode' : 'ClosedRay'
    }
    
    return mk_request(cmd_, params_, filename=f'trades/{ticker}', isV1=True)
     

def get_session_info()->None:
    cmd_ = "getSidInfo"
    params_ = {}
    
    mk_request(cmd_, params_, filename="sid_info") 


def get_trades()->None:
    """
    Получении истории сделок по пользователю
    """
    cmd_ ='getTradesHistory'
    params_ = {
        "nt_ticker": "SBER",
    }

    mk_request(cmd_, params_)


def get_sec_info(ticker: str)->None:
    cmd_ ='getSecurityInfo'
    params_ = {
        'ticker': f'{ticker}',
        'sup': True
    }
    mk_request(cmd_, params_)


def get_sec_data(ticker)->None:
    cmd_ ='getStockData'
    params_ = {
        'ticker': f'{ticker}',
        'lang': 'ru'
    }
    mk_request(cmd_, params_)
    

def auth()->None:
    cmd_ = "getSecuritySms"
    mk_request(cmd_)

    sms_code = input("Введи код из смс: ")
    sms_code = sms_code.strip()
    
    cmd_ = "openSecuritySession"

    params_ = {
        "safetyTypeId": 3,
        "validationKey": f"{sms_code}"
    }
    
    mk_request(cmd_, params_)
    
    
def make_df_from_json(df_json):
    df = pd.DataFrame()

    if 'error' in df_json.keys() or len(df_json)<2:
        return df

    for ticker in df_json['hloc'].keys():
        hloc = np.array(df_json['hloc'][ticker])
        l, _ = hloc.shape
        vl = df_json['vl'][ticker]
        tradedates = df_json['xSeries'][ticker]
        high_price = hloc[:,0]
        low_price = hloc[:,1]
        open_price = hloc[:,2]
        close_price = hloc[:,3]

        dataset = {
            'tradedate': tradedates,
            'ticker': [ticker] * l,
            'short_name': [df_json['info'][ticker]['short_name']] * l, 
            'currency': [df_json['info'][ticker]['currency']] * l, 
            'high_price': high_price,
            'low_price': low_price,
            'open_price': open_price,
            'close_price': close_price,
            'volume': vl
        }

        tmp = pd.DataFrame(dataset).sort_values('tradedate')
        tmp['tradedate'] = pd.to_datetime(tmp['tradedate'], unit='s') 
        df = pd.concat([df, tmp])
    
    return df


def download_bonds(bonds: List[str]=None) -> None:
    # bonds = ['KZ_06_4410', 'MUM132_0007']
    all_df = []
    for bond in bonds:
        search_res = search_ticker(bond)['found']
        ticker = search_res[0]['t'] if len(search_res)>0 else f'{bond}.KZ'
        path_trades = Path(f'output/trades/{ticker}.json')
        if not path_trades.exists():
            print(f'downloading {bond} from Tradernet')
            df_json = get_trade_hist(ticker,
                        from_=datetime(2019, 1, 1),
                        to_=datetime(2022, 12, 31), timeframe=1440)
        else:
            with open(path_trades, 'r', encoding='utf8') as f:
                df_json = json.load(f) 
        df = make_df_from_json(df_json)
        # print(df)
        # print(ticker, df.shape)
        if df.shape[0] !=0:
            all_df.append(df)
    return pd.concat(all_df, ignore_index=True)
 

def read_list_gov_bond():
    df = pd.read_csv('list_gov_bonds_KZ.txt', sep='\t', encoding='cp1251')
    return df.loc[df['Ticker'].notnull(),'Ticker'].values

if __name__=="__main__":
    # auth()
    # get_trades()
    # print(search_ticker("MUM132_0007")['found'][0])
    # get_sec_data("AAPL")
    # get_session_info()
    # get_user_data() 
    # get_trade_hist("MUM132_0007.KZ",
    #                from_=datetime(2019, 1, 1),
    #                to_=datetime(2022, 12, 31))
    # download_bonds(['AT_01_2006', 'KZ_05_2410', 'KZ_06_4410', 'KZ_07_2507', 'KZ_22_4507',
    #  'MOM036_0091', 'MOM060_0052', 'MUM084_0017', 'MUM096_0012', 'MUM108_0013',
    #  'MUM120_0016', 'MUM120_0018', 'MUM132_0005', 'MUM132_0006', 'MUM144_0003',
    #  'MUM144_0009', 'MUM156_0002', 'MUM156_0005', 'MUM156_0006', 'MUM168_0003',
    #  'MUM168_0005', 'MUM180_0011', 'MUM180_0012', 'MUM240_0001', 'OM_01_2908',
    #  'TR_01_2408', 'TR_02_2904', 'US_04_2908'])
    l = read_list_gov_bond()[:10]
    print(l)
    print(download_bonds(l))
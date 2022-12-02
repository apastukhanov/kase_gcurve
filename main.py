from datetime import datetime
import math
from typing import List

from functools import partial

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

import pandas as pd

import requests

from config import HEADERS



def download_gcurve_params()->None:
    URL = "https://kase.kz/ru/documents/curve"
    r = requests.get(URL)
    with open("output.xls", "wb") as f:
        f.write(r.content)


def get_tradedates()->List:
    df = get_df_with_params()
    return df['tradedate'].sort_values(ascending=True)


def get_df_with_params()->pd.DataFrame:
    df = pd.read_excel('output.xls')
    df = df.iloc[3:].reset_index(drop=True)

    df.columns = ['tradedate', 'B0', 'B1', 'B2', 'TAU']

    df['tradedate'] = pd.to_datetime(df['tradedate'], format='%Y-%m-%d %H:%M:%S')
    df[['B0','B1','B2','TAU']] = df[['B0','B1','B2','TAU']].astype('float64') 
    return df


def get_gcurve_yield(tradedate: datetime, m: float)->float:
    '''
    #print(datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S"))
    #download_gcurve_params()
    # m = 5.003
    # m = 5.336 #11.387
    # m = 5.392 #0.112805
    # print(get_gcurve_yield(datetime(2022,11,23), m))
    '''
    df = get_df_with_params()
    row = df.loc[df['tradedate']==tradedate]
    return (row["B0"] \
        +((row["B1"]+row["B2"])*(row["TAU"]/m)*(1-math.exp(-m/row["TAU"]))) \
        -row["B2"]*math.exp(-m/row["TAU"])).values[0]


def plot_gcurve(tradedate:datetime, flag_renew_plot=True)->None:

    dur = [0.25, 0.5, 0.75, 1, 2, 3, 5.392, 10, 15, 20, 25, 30]
    tradedates = [tradedate for i in range(len(dur))]
    values = list(map(get_gcurve_yield, tradedates, dur))

    plt_df = pd.DataFrame.from_dict({'duration':dur, 'yield': values})

    if flag_renew_plot:
        plt.clf()

    g = sns.lineplot(data=plt_df, 
        x='duration', 
        y='yield', legend='brief', label =f"{tradedate.day}-{tradedate.month}-{tradedate.year}").set(title=datetime.strftime(tradedate, "G-curve as for %d.%m.%Y"))

    plt.legend(loc='best')

    plt.show()


def animate_gcurve():
    fig = plt.figure()

    anim = FuncAnimation(fig, partial(plot_gcurve, flag_renew_plot=False), frames = get_tradedates().iloc[-10:],
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
        f.write(r.text)
    df = pd.read_html(r.text)[0]
    df = df.loc[~(df.iloc[:,1].str.lower().str.contains('итого'))]
    df['tradedate'] = tradedate
    df.to_csv(f'trades/{market}_{price_filter}_{tradedate}.csv', index=False, sep=';')
    return df


def parse_sec_info(isin: str):
    url = f"https://kase.kz/ru/gsecs/show/{isin}/"
    s = requests.session()
    s.headers.update(HEADERS)
    r = s.get(url)
    with open(f'sec_info_{isin}.html', 'w', encoding='utf8') as f:
        f.write(r.text)
    
    # print(pd.read_html(r.text)[-1])



def create_df_from_params() -> pd.DataFrame:
    params = get_df_with_params()
    dur = [0.25, 0.5, 0.75, 1, 2, 3, 5, 10, 15, 20, 25, 30]

    df = pd.DataFrame()

    def calc_g_curve(row, m):
        return (row["B0"] \
            +((row["B1"]+row["B2"])*(row["TAU"]/m)*(1-math.exp(-m/row["TAU"]))) \
            -row["B2"]*math.exp(-m/row["TAU"]))

    for num, row in params.iterrows():
        tradedate = row['tradedate']
        rows = [row for i in range(len(dur))]
        values = list(map(calc_g_curve, rows, dur))
        tmp_dict = {'tradedate': tradedate}
        tmp_dict.update({k:v for k, v in zip(dur, values) })
        df = pd.concat([df, pd.DataFrame(tmp_dict, index=[0])], ignore_index=True)

    return df


def create_df_from_params_vect():
    params = get_df_with_params()
    dur = [0.25, 0.5, 0.75, 1, 2, 3, 5, 10, 15, 20, 25, 30]
    df = pd.DataFrame(columns=[f'{d}' for d in dur])

    def calc(row, m):
        return (row["B0"] \
            +((row["B1"]+row["B2"])*(row["TAU"]/m)*(1-math.exp(-m/row["TAU"]))) \
            -row["B2"]*math.exp(-m/row["TAU"]))

    for d in dur:
        df[f'{d}'] = params.apply(partial(calc, m=d), axis=1)
    return pd.concat([params['tradedate'], df], axis=1)


if __name__=='__main__':
    # download_gcurve_params()
    # animate_gcurve()
    # parse_trades('16.11.2022', 'kase-global', '')
    parse_sec_info('KZ_06_4410')
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
    # print(f'func speed is {toc-tic:0.4f} seconds')
from dataclasses import dataclass

from typing import List
from datetime import datetime
from dateutil import parser

from scipy import optimize
import pandas as pd
import numpy as np

from main import parse_sec_info, parse_trades


@dataclass
class Bond:
    face_value: float
    coupon:float
    bond_price:float
    freq: float
    periods: List[datetime] 

    def get_price(self, rate):
        total_coupons_pv = self.get_coupons_pv(rate)
        face_value_pv = self.get_face_value_pv(rate)
        result = total_coupons_pv + face_value_pv
        return result

    def get_duration(self, rate, as_for_date: datetime):
        cash_flow = np.array([self.coupon/(1+rate/self.freq)**period
                              for period, _ in enumerate(self.periods, start=1)])
        nominal_pv = self.get_face_value_pv(rate)
        cash_flow[-1] += nominal_pv
        # print(cash_flow)
        period_dist = np.array([(period - np.datetime64(as_for_date)
                                 ).astype('timedelta64[D]') / np.timedelta64(1, 'D')
                                for period in self.periods])
        # print(period_dist)
        dur = sum(period_dist * cash_flow)/sum(cash_flow)
        # print(dur)
        return dur
    
    def get_coupons_pv(self, rate):
        cash_flow = [self.coupon/(1+rate/self.freq)**period for period, _ in enumerate(self.periods, start=1) ]
        # rates_l = [period for period, _ in enumerate(self.periods, start=1) ]
        # print(f'{rates_l=}')
        # print(f'{sum(cash_flow)=}')
        return sum(cash_flow)

    def get_face_value_pv(self, rate):
        periods = len(self.periods)
        fvpv = self.face_value / (1+rate/self.freq)**periods
        # print(f'{fvpv=}')
        return fvpv

    def get_ytm(self,  estimate=0.01):
        get_yield = lambda rate: self.get_price(rate) - self.bond_price
        return round(optimize.newton(get_yield, estimate),4)
    
    @classmethod
    def find_bond(cls, code: str,
                       bond_price:float, 
                       rep_date: datetime):
        bond_df = parse_sec_info(code)
        if bond_df is None:
            return None

        bond_df['Дата начала купонной выплаты'] = pd.to_datetime(bond_df['Дата начала купонной выплаты'],
                                                                 format='%d.%m.%Y')
        # print(bond_df.head().columns, bond_df['ISIN'].values[0])
        coupon = bond_df['Ставка, % год.'].iloc[0]
        face_value = 100
        bond_price = bond_price 
        bond_df = bond_df.loc[bond_df['Дата начала купонной выплаты']>rep_date]
        periods = bond_df['Дата начала купонной выплаты'].values 
        
        return Bond(face_value=face_value, bond_price=bond_price, 
                    coupon=coupon, freq=1, periods=periods)


if __name__=="__main__":
    # print(get_ytm(bond_price=95.05, face_value=100, coupon=5.75, years=2, freq=1))

    # df = pd.read_csv('sec_info/KZ_06_4410.csv', parse_dates=True)
    # df = parse_sec_info("KZ_06_4410")
    df = parse_trades('02.12.2022', 'gsecs', '#gsec_clean')
    bonds = df.loc[df['Объем, млн KZT']>0][['Код', 'Цена последней сделки']].values
    
    for kod, price in bonds:
    
        b = Bond.find_bond(code=kod, bond_price=price, rep_date=datetime(2022,12,2))
        y = b.get_ytm()
        # print(kod, y, price, b.get_duration(y, datetime(2022,12,2)))

    # df['Дата начала купонной выплаты'] = pd.to_datetime(df['Дата начала купонной выплаты'], format='%d.%m.%Y')
    # coupon = df['Ставка, % год.'].iloc[0]
    # face_value = 100
    # bond_price = 98.9999
    # freq = df['freq'].iloc[0]
    # df = df.loc[df['Дата начала купонной выплаты']>datetime(2022,12,2)]
    # periods = df['Дата начала купонной выплаты'].values 
    
    # b = Bond(face_value=face_value, bond_price=bond_price, 
    #    coupon=coupon, freq=1, periods=periods)
    
    # print(b)
    # print(len(periods))
    # print(b.get_price(rate=0.0986))
    # print(b.get_ytm()) #0.049360998076785484
    # ['tradedate', 
    #  'Код', 'Режим','Минимальная цена','Максимальная цена', 
    #  'Дни до погашения',  'Цена первой сделки', 'Цена последней сделки', 
    #  'Количество сделок', 'Объем, млн KZT', 'Средневзвеш. цена', 'Yield, %']
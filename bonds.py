from dataclasses import dataclass

from typing import List
from datetime import datetime

from scipy import optimize
import pandas as pd


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
    
    def get_coupons_pv(self, rate):
        cash_flow = [self.coupon/(1+rate/self.freq)**period for period, _ in enumerate(self.periods, start=1) ]
        rates_l = [period for period, _ in enumerate(self.periods, start=1) ]
        print(f'{rates_l=}')
        print(f'{sum(cash_flow)=}')
        return sum(cash_flow)

    def get_face_value_pv(self, rate):
        periods = len(self.periods)
        fvpv = self.face_value / (1+rate/self.freq)**periods
        print(f'{fvpv=}')
        return fvpv

    def get_ytm(self,  estimate=0.01):
        get_yield = lambda rate: self.get_price(rate) - self.bond_price
        return optimize.newton(get_yield, estimate)
    


def get_price(coupon, face_value, rate, years, freq=1):
    total_coupons_pv = get_coupons_pv(coupon, rate, years, freq)
    face_value_pv = get_face_value_pv(face_value, rate, years)
    result = total_coupons_pv + face_value_pv
    return result


def get_face_value_pv(face_value, rate, years):
    fvpv = face_value / (1+rate)**years
    return fvpv


def get_coupon_pv(coupon, rate, period, freq):
    pv = coupon / (1 + rate/freq)**period
    return pv


def get_coupons_pv_with_periods(coupon, rate, periods, freq=1):
    cash_flow = [coupon/(1+rate/freq)**period for period, _ in enumerate(periods, start=1) ]
    print(cash_flow)
    return sum(cash_flow)


def get_coupons_pv(coupon, rate, years, freq=1):
    pv = 0
    for period in range(years*freq):
        pv += get_coupon_pv(coupon, rate, period+1, freq)
    return pv 


    
def get_ytm(bond_price, face_value, coupon, years, freq=1, estimate=0.01):
    get_yield = lambda rate: get_price(coupon, face_value, rate, years, freq) - bond_price
    return optimize.newton(get_yield, estimate)


if __name__=="__main__":
    # print(get_ytm(bond_price=95.05, face_value=100, coupon=5.75, years=2, freq=1))
    from main import parse_sec_info
    
    # df = pd.read_csv('sec_info/KZ_06_4410.csv', parse_dates=True)
    df = parse_sec_info("KZ_06_4410")
    # df['Дата начала купонной выплаты'] = pd.to_datetime(df['Дата начала купонной выплаты'], format='%d.%m.%Y')
    coupon = df['Ставка, % год.'].iloc[0]
    face_value = 100
    bond_price = 98.9999
    freq = df['freq'].iloc[0]
    df = df.loc[df['Дата начала купонной выплаты']>datetime(2022,12,2)]
    periods = df['Дата начала купонной выплаты'].values 
    
    b = Bond(face_value=face_value, bond_price=bond_price, 
       coupon=coupon, freq=1, periods=periods)
    
    print(b)
    print(len(periods))
    # print(b.get_price(rate=0.0986))
    print(b.get_ytm())
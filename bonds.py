from dataclasses import dataclass

from typing import List
from datetime import datetime
from dateutil import parser

from scipy import optimize
import pandas as pd
import numpy as np

from main import parse_trades, parse_sec_info


@dataclass
class Bond:
    face_value: float
    coupon:float
    bond_price:float
    freq: float
    periods: List[datetime] 
    mat_date: datetime = None
    as_for_date: datetime = None
    isin: str = None

    def get_price(self, rate):
        total_coupons_pv = self.get_coupons_pv(rate)
        face_value_pv = self.get_face_value_pv(rate)
        result = total_coupons_pv + face_value_pv
        return result

    def get_duration(self, rate):
        as_for_date = self.as_for_date
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
        as_for_date = self.as_for_date
        period_dist = np.array([(period - np.datetime64(as_for_date) 
                                 ).astype('timedelta64[D]') / np.timedelta64(1, 'D')
                                for period in self.periods])
        # cash_flow = [self.coupon/(1+rate/self.freq)**period for period, _ in enumerate(self.periods, start=1) ]
        cash_flow = self.coupon/ (1 + rate/ self.freq) ** (period_dist/365) 
        # rates_l = [period for period, _ in enumerate(self.periods, start=1) ]
        # print(f'{rates_l=}')
        # print(f'{sum(cash_flow)=}')
        return np.sum(cash_flow)

    def get_face_value_pv(self, rate):
        periods = len(self.periods)
        rep_date = np.datetime64(self.as_for_date)
        mat_date = np.datetime64(self.mat_date)
        # if not self.mat_date:
        fvpv = self.face_value / (1+rate/self.freq)**(((mat_date - rep_date) / np.timedelta64(1, 'D'))/365)
    # print(f'{fvpv=}')
        return fvpv
        # print("no mat date=", self)
        # return self.face_value / (1 + rate/self.freq)

    def get_ytm(self,  estimate=0.01):
        get_yield = lambda rate: self.get_price(rate) - self.bond_price
        return round(optimize.newton(get_yield, estimate),4)
    
    def get_fix_days_before_mat(self, as_for_date:datetime):
        if self.mat_date:
            return (self.mat_date - as_for_date).days
        return 9999
    
    @classmethod
    def find_bond(cls, code: str,
                       bond_price:float, 
                       rep_date: datetime):
        bond_df = parse_sec_info(code)

        if bond_df is None or bond_df.shape[0]==0:
            return None
        
        if 'Дата начала купонной выплаты' in bond_df.columns:
            bond_df['Дата начала купонной выплаты'] = pd.to_datetime(bond_df['Дата начала купонной выплаты'],
                                                                    format='%d.%m.%Y')
        else: 
            bond_df['Дата начала купонной выплаты'] = None 
        # print(bond_df.columns)
        if 'Дата погашения' in bond_df.columns:
            bond_df['Дата погашения'] = bond_df["Дата погашения"].apply(lambda x: 
                                                f"{x.split('.')[0]}.{x.split('.')[1]}.20{x.split('.')[2]}")
            bond_df['Дата погашения'] = pd.to_datetime(bond_df['Дата погашения'],
                                                                    format='%d.%m.%Y')
            mat_date = bond_df['Дата погашения'].iloc[0] 
            
            if  bond_df['Дата начала купонной выплаты'].iloc[-1] is None \
                and bond_df['Дата начала купонной выплаты'].count() < 2:
                    bond_df['Дата начала купонной выплаты'] = mat_date 
        elif 'Период погашения' in bond_df.columns:
            bond_df['Период погашения'] = bond_df["Период погашения"].apply(lambda x: 
                                                f"{x.split('.')[0]}.{x.split('.')[1]}.20{x.split('.')[2]}")
            bond_df['Период погашения'] = pd.to_datetime(bond_df['Период погашения'],
                                                                    format='%d.%m.%Y')
            mat_date = bond_df['Период погашения'].iloc[0] 
        else:
            mat_date = None
        # print(bond_df.head().columns, bond_df['ISIN'].values[0])
        # if not 'Ставка, % год.' in bond_df.columns:
        #     return Bond(face_value=100, bond_price=bond_price, coupon=0, 
        #                 freq=1, periods=None, mat_date=mat_date)
        if len(set(bond_df.columns) & set(['Ставка, % год.', 'Текущая купонная ставка, % годовых'])) > 0:
            try: 
                coupon = bond_df['Ставка, % год.'].iloc[0]
            except Exception as e:
                coupon = bond_df['Текущая купонная ставка, % годовых']
        else: 
            coupon = 0
        face_value = 100
        col_nkd =bond_df.loc[bond_df['Дата начала купонной выплаты']<rep_date, 'Дата начала купонной выплаты'].sort_values() 
        if col_nkd.count() == 0:
            nkd = 0
        elif col_nkd.count()< 2:
            d = np.datetime64(col_nkd.values[0])
            days_diff = (np.datetime64(rep_date) - d).astype('timedelta64[D]') / np.timedelta64(1, 'D')
            nkd = (days_diff/ 365) * coupon
        else:
            nkd = ((rep_date - col_nkd.iloc[-1]).days/ 365) * coupon
        bond_price = bond_price + nkd 
        bond_df = bond_df.loc[bond_df['Дата начала купонной выплаты']>rep_date]
        periods = bond_df['Дата начала купонной выплаты'].values
        
        return Bond(face_value=face_value, bond_price=bond_price, 
                    coupon=coupon, freq=1, periods=periods, mat_date=mat_date, as_for_date=rep_date,
                    isin = code)


def test_ytm():
    df = parse_trades('02.12.2022', 'gsecs', '#gsec_clean')
    bonds = df.loc[df['Объем, млн KZT']>0][['Код', 'Цена последней сделки']].values
    
    for kod, price in bonds:
    
        b = Bond.find_bond(code=kod, bond_price=price, rep_date=datetime(2022,12,2))
        y = b.get_ytm()
        d = b.get_fix_days_before_mat(datetime(2022,12,2))
        print(kod, y, price, b.get_duration(y, datetime(2022,12,2)))
        df.loc[df['Код']==kod,'Yield, %'] = y
        df.loc[df['Код']==kod,'Дни до погашения'] = d
    
    # df.to_clipboard()






if __name__=="__main__":
    # test_ytm()
    # print(get_ytm(bond_price=95.05, face_value=100, coupon=5.75, years=2, freq=1))

    # df = pd.read_csv('sec_info/KZ_06_4410.csv', parse_dates=True)
    # df = parse_sec_info("KZ_06_4410")

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
    # plot_gcurve(datetime(2022,12,5))
    b = Bond.find_bond('MOM048_0054', bond_price=105.8757, rep_date=datetime(2022, 12, 8))
    print(b)
    print(b.get_ytm())
    print(b.get_fix_days_before_mat(datetime(2022,12,9)))
    # print(b.get_ytm())
    # test_ytm()
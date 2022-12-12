from datetime import datetime
from dataclasses import dataclass

import math

import pandas as pd
import numpy as np

from scipy import optimize

from nelson_siegel_svensson.calibrate import calibrate_ns_ols, NelsonSiegelCurve

from matplotlib import pyplot as plt
import seaborn as sns

from main import parse_trades, get_gcurve_yield
from bonds import Bond



EPS = np.finfo(float).eps


@dataclass
class Gcurve:
    beta0: float
    beta1: float
    beta2: float
    tau: float
    
    def get_factors(self, T: np.ndarray):    
        tau = self.tau
        zero_idx = T <= 0
        T[zero_idx] = EPS
        exp_tt0 = np.exp(-T / tau)
        factor1 = (1 - exp_tt0) / (T / tau)
        factor2 = factor1 - exp_tt0 
        T[zero_idx] = 0
        factor1[zero_idx] = 1
        factor2[zero_idx] = 0
        return factor1, factor2
    
    def get_factors_matrix(self, T: np.ndarray):
        factor1, factor2 = self.get_factors(T)
        return np.stack([np.ones(T.size), factor1, factor2]).transpose()

    def get_gcurve_points(self, T: np.ndarray):
        factor1, factor2 = self.get_factors(T)
        return self.beta0 + self.beta1 * factor1 + self.beta2 * factor2
    
    def get_yield_t(self, T: np.ndarray):
        curve_points = self.get_gcurve_points(T)
        return 100 * (np.exp(curve_points/100)-1)
    
      


def find_gcurve_params(t, y):
    t = np.array(t)
    y = np.array(y)

    curve, status = calibrate_ns_ols(t, y, tau0=1.0)
    assert status
    print(curve)


def plot_gcurve(tradedate: datetime, flag_renew_plot=True) -> None:
    list_dur = [0.25, 0.5, 0.75, 1, 2, 3, 5, 10, 15, 20, 25, 30]
    list_td = [tradedate for i in range(len(list_dur))]
    values = list(map(get_gcurve_yield, list_td, list_dur))

    plt_df = pd.DataFrame.from_dict({'duration': list_dur,
                                     'yield': values})

    trades = parse_trades(f'{tradedate.strftime("%d.%m.%Y")}', 'gsecs', '#gsec_clean').iloc[:-1]
    bonds = trades[['Код', 'Цена последней сделки']].values

    for kod, price in bonds:
        b = Bond.find_bond(code=kod, bond_price=price, rep_date=tradedate)
        if b is None:
            continue
        r = b.get_ytm()
        trades.loc[trades["Код"] == kod, ['Yield, %']] = round(r, 4)
        trades.loc[trades["Код"] == kod, ['Дни до погашения']] = \
            int(b.get_duration(r, tradedate)) / 365

    if flag_renew_plot:
        plt.clf()

    sns.lineplot(data=plt_df,
                     x='duration',
                     y='yield', legend='brief',
                     label=f"{tradedate.day}-{tradedate.month}-{tradedate.year}").set(
        title=datetime.strftime(tradedate, "G-curve as for %d.%m.%Y"))

    sns.scatterplot(data=trades[['Дни до погашения', 'Yield, %']],
                    x='Дни до погашения',
                    y='Yield, %',
                    marker="^",
                    color="red")

    plt.legend(loc='best')
    plt.show()



def get_betas(tau, target_y, T):
    curve = Gcurve(0,0,0,tau)
    m = curve.get_factors_matrix(T)
    betas = np.linalg.lstsq(m, target_y, rcond=None)[0]
    curve = Gcurve(betas[0], betas[1], betas[2], tau)
    return curve, betas


def min_square(tau, target_y, T):
    curve, betas = get_betas(tau, target_y, T)
    y_t = curve.get_yield_t(T)
    return np.sum((y_t - target_y)**2)


def find_yeild(y, t, tau0):
    res = optimize.minimize(min_square, x0=tau0, args=(y, t))
    curve, betas = get_betas(res.x[0], y, t)
    return curve
    


if __name__ == '__main__':
    plot_gcurve(datetime(2022, 12, 7))
    # dur = [0.25, 0.5, 0.75, 1, 2, 3, 5, 10, 15, 20, 25, 30]
    # ns = NelsonSiegelCurve(beta0=0.093, beta1=0.055, beta2=0.0, tau=1.2)
    # dur = np.array(dur)
    # print(ns.factors(dur)[0])
    # print(ns.factors(dur)[1])
    # print(np.exp(-dur/ns.tau))
    # l = np.arange(1,11)
    # print(l)
    # r = 0.01
    # f1 = np.round(1/((1+r)**l), 4)
    # # print(f1)
    t = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    y = np.array([0.01, 0.011, 0.013, 0.016, 0.019, 0.021, 0.026, 0.03, 0.035, 0.037, 0.038, 0.04])
    curve, status = calibrate_ns_ols(t, y, tau0=1.0)
    print(curve)
    print(find_yeild(y,t, tau0=1.0))
    # # NelsonSiegelCurve(beta0=0.042017393872432876, beta1=-0.031829031623813654, beta2=-0.02679731950812892, tau=1.7170972824332638)
    # Gcurve(b0=0.04201739390445989, b1=-0.031829031672107905, b2=-0.02679731926748868, tau=array([1.7170973]))

    

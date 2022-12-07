from datetime import datetime
from dataclasses import dataclass

import math

import pandas as pd
import numpy as np

from scipy import optimize

from nelson_siegel_svensson.calibrate import calibrate_ns_ols

from matplotlib import pyplot as plt
import seaborn as sns

from main import parse_trades, get_gcurve_yield
from bonds import Bond


@dataclass
class Gcurve:
    b0: float
    b1: float
    b2: float
    tau: float

    def get_gcurve_point(self, m):
        return self.b0 \
               + ((self.b1 + self.b2) * (self.tau / m) * (1 - math.exp(-m / self.tau))) \
               - self.b2 * math.exp(-m / self.tau)


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


if __name__ == '__main__':
    plot_gcurve(datetime(2022, 12, 1))

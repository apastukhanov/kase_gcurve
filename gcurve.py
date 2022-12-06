from dataclasses import dataclass

import math

import pandas as pd
import numpy as np

from scipy import optimize

from nelson_siegel_svensson.calibrate import calibrate_ns_ols


@dataclass
class Gcurve:
    b0: float
    b1: float
    b2: float
    tau: float
    
    
    def get_gcurve_point(self, m):
        return self.b0 \
                +((self.b1+self.b2)*(self.tau/m)*(1-math.exp(-m/self.tau))) \
                -self.b2*math.exp(-m/self.tau)
                
     

def find_gcurve_params(t,y):
    t = np.array(t)
    y = np.array(y)
    
    curve, status = calibrate_ns_ols(t, y, tau0=1.0)
    assert status
    print(curve)
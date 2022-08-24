import pandas as pd
import numpy as np
import statsmodels.api as sm
import itertools

def _interval(start, stop, size):

    increment_by = (stop - start)//size
    cur = start
    yield start
    for increment in itertools.cycle((increment_by, increment_by)):
        cur += increment
        if cur >= stop:
            break
        yield cur
    yield stop


def _find_tvalue(df):
    x = list(df.index)
    y = df.close
    x = sm.add_constant(x)

    model = sm.OLS(y, x)
    r = model.fit()

    return r.tvalues[-1]


class TrendScanning:

    def __init__(self, df: pd.DataFrame, look_forward: int):

        self.df = df.reset_index(drop=True)
        self.look_forward = look_forward

        self.df['t_val_aux'] = 0
        self.df['t_val'] = 0

        inter = list(_interval(0, len(self.df)-1, self.look_forward))

        for i in range(0, len(inter)-1):

            inside_interval = self.df.index.isin(range(inter[i], inter[i+1]))

            self.df.t_val_aux = np.where(inside_interval,
                                         _find_tvalue(self.df[inside_interval]),
                                         self.df.t_val_aux)

            self.df.t_val = np.where(np.abs(self.df.t_val) < np.abs(self.df.t_val_aux),
                                     self.df.t_val_aux,
                                     self.df.t_val)

        self.df['sign'] = self.df.t_val/np.abs(self.df.t_val)

        self.trend_scan = self.df[['t_val', 'sign']]

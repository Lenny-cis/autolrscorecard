# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 23:02:10 2021

@author: Lenny
"""

import pandas as pd
import numpy as np
import copy

a = pd.DataFrame(np.random.random((10, 2)))
b = a.copy().values
c = np.mat(b)


def a1(x):
    x = x.copy()
    x.loc[1, :] = x.loc[0: 2, :].sum(axis=0)
    x.drop(0, axis=0, inplace=True)
    return x


def a2(x):
    x = x.copy()
    x[1] = x[0: 2].sum(axis=0)
    x = np.delete(x, 0, axis=0)
    return x


def a3(x):
    x = copy.copy(x)
    x[1, :] = x[0, :] + x[1, :]
    x = np.delete(x, 0, axis=0)
    return x


[a1(a) for _ in range(10)]
[a2(b) for _ in range(10)]
[a3(c) for _ in range(10)]

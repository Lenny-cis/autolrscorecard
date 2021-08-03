# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:33:13 2021

@author: linjianing
"""

import os
import numpy as np
import pandas as pd
import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), r'..\..\..\..'))
sys.path.append(root)
from autolrscorecard.utils.woebin_utils import merge_lowpct_zero

arr = np.array([[9000, 16000, 21000, 24000, 25000, 20000, 900],
                [1000, 4000, 9000, 16000, 25000, 30000, 100]]).T
arr_na = np.array([[3000, 500]])
cut = [-np.inf, 10, 20, 35, 40, 60, 80, np.inf]
a, b = merge_lowpct_zero(arr, cut)
print('a', a)
print('b', b)
arr = np.array([[9000, 16000, 21000, 24000,40, 25000, 20000],
                [1000, 4000, 9000, 16000, 60, 25000, 30000]]).T
arr_na = np.array([[3000, 500]])
a, b = merge_lowpct_zero(arr, cut)
print('a', a)
print('b', b)
arr = np.array([[9000, 16000, 210, 24000, 0, 25000, 20000],
                [1000, 4000, 90, 16000, 40, 25000, 30000]]).T
a, b = merge_lowpct_zero(arr, cut)
print('a', a)
print('b', b)

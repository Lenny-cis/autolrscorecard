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
from autolrscorecard.utils.woebin_utils import merge_lowpct, best_merge_by_idx

arr = np.array([[9000, 16000, 21000, 24000, 60, 25000, 20000, 900],
                [1000, 4000, 9000, 16000, 40, 25000, 30000, 100]]).T
arr_na = np.array([[3000, 500]])
a, b = merge_lowpct(arr, 0.1)
print('a', a)
print('b', b)
c, d = best_merge_by_idx(arr, 1)
print('c', c)
print('d', d)
c, d = best_merge_by_idx(arr, 4)
print('c', c)
print('d', d)
c, d = best_merge_by_idx(arr, 7)
print('c', c)
print('d', d)
arr = np.array([[9000, 16000, 21000, 24000,40, 25000, 20000, 900],
                [1000, 4000, 9000, 16000, 60, 25000, 30000, 100]]).T
arr_na = np.array([[3000, 500]])
a, b = merge_lowpct(arr, 0.1)
print(a)
print(b)

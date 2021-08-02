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
from autolrscorecard.utils.woebin_utils import merge_zeronum

arr = np.array([[9000, 16000, 21000, 24000, 0, 25000, 20000, 900],
                [1000, 4000, 9000, 16000, 40, 25000, 30000, 100]]).T
arr_na = np.array([[3000, 500]])
a, b = merge_zeronum(arr)
print('a', a)
print('b', b)
arr = np.array([[9000, 16000, 21000, 24000,0, 25000, 20000, 900],
                [1000, 4000, 0, 16000, 60, 25000, 30000, 100]]).T
arr_na = np.array([[3000, 500]])
a, b = merge_zeronum(arr)
print(a)
print(b)

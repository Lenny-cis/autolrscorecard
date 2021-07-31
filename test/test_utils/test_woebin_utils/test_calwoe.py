# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:09:10 2021

@author: linjianing
"""

import sys
import os
import numpy as np
root = os.path.abspath(os.path.join(os.path.dirname(__file__), r'..\..\..\..'))
sys.path.append(root)
from autolrscorecard.utils.woebin_utils import calwoe

arr = np.array([[9000, 16000, 21000, 24000, 25000, 20000],
                [1000, 4000, 9000, 16000, 25000, 30000]]).T
arr_na = np.array([[3000, 500]])
a = calwoe(arr, arr_na)

arr = np.array([[9000, 16000, 21000, 24000, 25000, 20000],
                [1000, 4000, 9000, 16000, 25000, 30000]]).T
arr_na = np.array([[3000, 0]])
a1 = calwoe(arr, arr_na)

arr = np.array([[9000, 16000, 21000, 24000, 25000, 20000],
                [1000, 4000, 9000, 16000, 25000, 30000]]).T
arr_na = np.array([[0, 500]])
a2 = calwoe(arr, arr_na)

arr = np.array([[9000, 16000, 21000, 24000, 25000, 20000],
                [1000, 4000, 9000, 16000, 25000, 30000]]).T
arr_na = np.array([[0, 0]])
a3 = calwoe(arr, arr_na)

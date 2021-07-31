# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:33:13 2021

@author: linjianing
"""

import os
import numpy as np
import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), r'..\..\..\..'))
sys.path.append(root)
from autolrscorecard.utils.woebin_utils import merge_arr_by_idx

arr = np.array([[9000, 16000, 21000, 24000, 25000, 20000],
                [1000, 4000, 9000, 16000, 25000, 30000]]).T
arr_na = np.array([[3000, 500]])

a = merge_arr_by_idx(arr, [1])
a2 = merge_arr_by_idx(arr, [1, 3])
a3 = merge_arr_by_idx(arr, [2])

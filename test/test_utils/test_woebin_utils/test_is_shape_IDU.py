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
from autolrscorecard.utils.woebin_utils import(
    is_shape_I, is_shape_D, is_shape_U)

arr = np.array([9000, 16000, 21000, 24000, 25000, 200000])
a = is_shape_I(arr)
print('I == {}'.format(a))
arr = np.flip(arr)
a = is_shape_D(arr)
print('D == {}'.format(a))
arr = np.append(arr, np.flip(arr)[1:])
a = is_shape_U(arr)
print('U == {}'.format(a))

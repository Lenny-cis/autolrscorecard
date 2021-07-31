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
from autolrscorecard.utils.woebin_utils import (
    arr_badrate_shape, gen_badrate)

arr = np.array([[9000, 16000, 21000, 24000, 25000, 20000],
                [1000, 4000, 9000, 16000, 25000, 30000]]).T
arr_na = np.array([[3000, 500]])

print(gen_badrate(arr))

a = arr_badrate_shape(arr)
print('I == {}'.format(a))

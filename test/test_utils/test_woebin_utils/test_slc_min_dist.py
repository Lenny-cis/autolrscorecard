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
from autolrscorecard.utils.woebin_utils import (slc_min_dist)

arr = np.array([[9000, 16000, 21000, 24000, 30010, 20000],
                [1000, 4000, 9000, 16000, 19990, 30000]]).T
arr_na = np.array([[3000, 500]])

print(arr[:, 1]/arr.sum(axis=1))
a = slc_min_dist(arr)

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 19:54:08 2021

@author: linjianing
"""

import os
import numpy as np
import pandas as pd
import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), r'..\..\..\..'))
sys.path.append(root)
from autolrscorecard.utils.woebin_utils import cut_diff_ptp

cut = [-np.inf, 10, 20, 35, 40, 60, 80, np.inf]
qt = [-5, 0, 40, 70, 100]
print(cut)
a = cut_diff_ptp(cut, qt)
print(a)
qt = [-500, 0, 40, 70, 1000]
b = cut_diff_ptp(cut, qt)
print(b)

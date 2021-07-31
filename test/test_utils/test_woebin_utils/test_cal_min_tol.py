# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 10:57:47 2021

@author: linjianing
"""


import os
import numpy as np
os.chdir(os.path.join(os.path.dirname(__file__), r'..\..\..\..'))
from autolrscorecard.utils.woebin_utils import cal_min_tol

a = cal_min_tol(np.array([1, 2, 6, 20, 21]))
print(a)
a = cal_min_tol(np.array([1, 2, 6, 5.1, 20, 21]))
print(a)

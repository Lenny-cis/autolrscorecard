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
from autolrscorecard.utils.woebin_utils import cut_adjust

a = cut_adjust([-np.inf, 10, 20, 30, 40, 50, 60, np.inf], [2])
print(a)
a = cut_adjust([-np.inf, 10, 20, 30, 40, 50, 60, np.inf], [])
print(a)
a = cut_adjust([-np.inf, 10, 20, 30, 40, 50, 60, np.inf], [2, 4])
print(a)
b = cut_adjust({'a': 0, 'b': 0, 'c': 1, 'd': 2, 'e': 3, 'f': 4, 'g': 4,
                'h': 5}, [2])
print(b)
c = cut_adjust({'a': 0, 'b': 0, 'c': 1, 'd': 2, 'e': 3, 'f': 4, 'g': 4,
                'h': 5}, [2, 4])
print(c)

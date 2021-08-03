# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:58:27 2021

@author: linjianing
"""

import os
import numpy as np
import pandas as pd
import sys
import ray
from joblib import load
root = os.path.abspath(os.path.join(os.path.dirname(__file__), r'..\..\..\..'))
sys.path.append(root)
from autolrscorecard.woebin import VarBinning

ray.init()
data_file = os.path.join(root, 'autolrscorecard', 'test', 'data',
                         'test_sample.h5')
data = pd.read_hdf(data_file, 'data')

test_vb = VarBinning(cut_cnt=30)
test_vb.fit(data.iloc[:, 0], data.loc[:, 'flag'])
a = test_vb.bins_set

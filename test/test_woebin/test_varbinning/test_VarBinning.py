# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:58:27 2021

@author: linjianing
"""

import os
import numpy as np
import pandas as pd
import time
from joblib import load, dump
root = os.path.abspath(os.path.join(os.path.dirname(__file__), r'..\..\..\..'))
os.chdir(root)
from autolrscorecard.woebin import VarBinning
from autolrscorecard import alsc_parallel
alsc_parallel.close()
data_file = os.path.join(root, 'autolrscorecard', 'test', 'data',
                          'test_sample.h5')
data = pd.read_hdf(data_file, 'data')

test_vb = VarBinning(cut_cnt=50, thrd_n=150)
ta = time.time()
test_vb.fit(data.iloc[:, 0], data.loc[:, 'flag'])
a = test_vb.bins_set
print(time.time()-ta)

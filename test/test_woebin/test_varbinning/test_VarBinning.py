# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:58:27 2021

@author: linjianing
"""

import os
import sys
import numpy as np
import pandas as pd
import time
from joblib import load, dump
root = os.path.abspath(os.path.join(os.path.dirname(__file__), r'..\..\..\..'))
sys.path.append(root)
os.chdir(root)
from autolrscorecard.woebin import VarBinning
data_file = os.path.join(root, 'autolrscorecard', 'test', 'data',
                          'test_sample.h5')
data = pd.read_hdf(data_file, 'data')

test_vb = VarBinning(cut_cnt=50, thrd_n=140)
ta = time.time()
test_vb.fit(data.iloc[:, 0], data.loc[:, 'flag'])
a = test_vb.bins_set
print(time.time()-ta)
test_vb.fast_search_best()
d = test_vb.selected_best
e = test_vb.transform(data.iloc[:, 0])
test_vb.grid_search_best(slc_mthd=['IV'], bins_cnt=6)
test_vb.manual_fit(data.iloc[:, 0], data.loc[:, 'flag'],
                   [-np.inf, 0.92, 2.046, 2.57, np.inf])
mbest = test_vb.selected_best
test_vb.plot_selected()
test_vb.plot_best_bins(r'./plot')

dump(test_vb, r'./plot/test_vb.pkl')
test_vb = load(r'./plot/test_vb.pkl')

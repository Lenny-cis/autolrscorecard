# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 23:43:08 2021

@author: Lenny
"""

import ray
import pandas as pd
import numpy as np
from woebin.varbinning import VarBinning
import autolrscorecard.variable_types.variable as vtype
# ray.init()
# %%
x = pd.Series(np.random.normal(size=10000), name='x')
y = np.exp(-x + np.random.normal(size=10000)) / (np.exp(-x + np.random.normal(size=10000)) + 1)
y = y > 0.8
y.name = 'y'
z = pd.DataFrame(x).join(pd.DataFrame(y))

vb = VarBinning.remote(variable_shape='IDU', **{'variable_type': vtype.Summ(4)})
vb.fit.remote(x, y)
ray.get(vb.get_bins_set.remote())
# vb2 = VarBinning(variable_shape='I', kwargs={'variable_type': vtype.Summ})
# vb2.fit()

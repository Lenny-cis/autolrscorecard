# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:53:36 2021

@author: linjianing
"""


import pp
import pandas as pd
import numpy as np


def sum_rand(r):
    np.random.seed(r)
    rs = np.random.random_sample(10000)
    return rs.sum()


job_server = pp.Server(ppservers=('localhost',))
job_server.get_ncpus()
sum_pppp = [job_server.submit(func=sum_rand, args=(i,)) for i in range(10)]
for job in sum_pppp:
    print(job())

print(1)

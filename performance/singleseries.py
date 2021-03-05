# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:29:57 2021

@author: linjianing
"""


import numpy as np
import pandas as pd
from utils.performance_utils import caliv


def missing_ratio(ser):
    """计算缺失率."""
    return ser.isna().sum() / len(ser)


def concentration_ratio(ser):
    """计算集中度."""
    if all(ser.isna()):
        return np.nan
    return ser.value_counts().iloc[0] / ser.count()


def num_unique(ser):
    """不同值的个数."""
    return ser.nunique()


def psi(basic_ser, check_ser, cut):
    """计算psi."""
    if isinstance(cut, dict):
        basic_bin = basic_ser.map(cut)
        check_bin = check_ser.map(cut)
    if isinstance(cut, list):
        basic_bin = pd.cut(basic_ser, cut, labels=False)
        check_bin = pd.cut(check_ser, cut, labels=False)
    basic_group = basic_bin.value_counts(dropna=False)
    check_group = check_bin.value_counts(dropna=False)
    basic_group, check_group = basic_group.align(check_group, join='outer')
    cross = pd.DataFrame.from_dict({
        'base': basic_group, 'check': check_group}).fillna(0)
    return caliv(cross.values)

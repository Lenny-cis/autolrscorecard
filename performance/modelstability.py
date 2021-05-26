# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:52:24 2021

@author: linjianing
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from autolrscorecard.utils.woebin_utils import make_tqdm_iterator
from autolrscorecard.utils.performance_utils import(
    caliv, gen_cut, apply_cut_bin)

PBAR_FORMAT = "Elapsed: {elapsed} | Progress: {l_bar}{bar}"


def vars_bin_psi(train_df, test_df):
    """特征psi."""
    n_res = {}
    x_names = list(train_df.columns)
    for x_name in x_names:
        train_x_ser = train_df.loc[:, x_name]
        test_x_ser = test_df.loc[:, x_name]
        train_group = train_x_ser.value_counts(dropna=False)
        test_group = test_x_ser.value_counts(dropna=False)
        train_group, test_group = train_group.align(test_group, join='outer')
        cross = pd.DataFrame.from_dict({
            'base': train_group, 'check': test_group}).fillna(0)
        psi_value = caliv(cross)
        n_res.update({x_name: psi_value})
    return n_res


def score_psi(train_df, test_df):
    """分数psi."""
    var_type = np.number
    train_ser = train_df.iloc[:, 0]
    test_ser = test_df.iloc[:, 0]
    cut = gen_cut(train_ser, var_type)
    train_bin = apply_cut_bin(train_ser, cut, var_type)
    train_group = train_bin.value_counts(dropna=False)
    test_bin = apply_cut_bin(test_ser, cut, var_type)
    test_group = test_bin.value_counts(dropna=False)
    train_group, test_group = train_group.align(test_group, join='outer')
    cross = pd.DataFrame.from_dict({
        'base': train_group, 'check': test_group}).fillna(0)
    psi_value = caliv(cross)
    return {'score': psi_value}


def repeatkfold_performance(x_df, func, n_r, n_s=5):
    """重复s折r次交叉验证计算psi."""
    rkf = RepeatedKFold(n_splits=n_s, n_repeats=n_r)
    df_idxs = pd.Series(x_df.index.to_list())
    rkf_idxs = rkf.split(df_idxs)
    tqdm_options = {'bar_format': PBAR_FORMAT,
                    'total': n_s * n_r,
                    'desc': 'PSI'}
    res = {}
    n = 0
    with make_tqdm_iterator(**tqdm_options) as progress_bar:
        for train_idxs, test_idxs in rkf_idxs:
            train_df = x_df.loc[df_idxs.iloc[train_idxs], :]
            test_df = x_df.loc[df_idxs.iloc[test_idxs], :]
            n_res = func(train_df, test_df)
            res.update({n: n_res})
            n += 1
            progress_bar.update()
    return pd.DataFrame.from_dict(res, orient='index')
# def repeatkfold_psi(x_df, n_r, n_s=5):
#     """重复s折r次交叉验证计算psi."""
#     rkf = RepeatedKFold(n_splits=n_s, n_repeats=n_r)
#     df_idxs = pd.Series(x_df.index.to_list())
#     rkf_idxs = rkf.split(df_idxs)
#     x_names = list(x_df.columns)
#     tqdm_options = {'bar_format': PBAR_FORMAT,
#                     'total': n_s * n_r,
#                     'desc': 'PSI'}
#     res = {}
#     n = 0
#     with make_tqdm_iterator(**tqdm_options) as progress_bar:
#         for train_idxs, test_idxs in rkf_idxs:
#             n_res = {}
#             train_df = x_df.loc[df_idxs.iloc[train_idxs], :]
#             test_df = x_df.loc[df_idxs.iloc[test_idxs], :]
#             for x_name in x_names:
#                 train_x_ser = train_df.loc[:, x_name]
#                 test_x_ser = test_df.loc[:, x_name]
#                 train_group = train_x_ser.value_counts(dropna=False)
#                 test_group = test_x_ser.value_counts(dropna=False)
#                 train_group, test_group = train_group.align(test_group, join='outer')
#                 cross = pd.DataFrame.from_dict({
#                     'base': train_group, 'check': test_group}).fillna(0)
#                 psi_value = caliv(cross)
#                 n_res.update({x_name: psi_value})
#             res.update({n: n_res})
#             n += 1
#             progress_bar.update()
#     return pd.DataFrame.from_dict(res, orient='index')

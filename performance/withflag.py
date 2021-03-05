# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:59:48 2021

@author: linjianing
"""


import numpy as np
import pandas as pd
from utils.performance_utils import (gen_cross, gen_cut, caliv, apply_woe,
                                     calks, calauc)


def IV(ser, y, var_type):
    """变量IV."""
    cut = gen_cut(ser, var_type, n=20, mthd='eqqt', precision=4)
    cross, cut = gen_cross(ser, y, cut, var_type)
    iv = caliv(cross)
    return iv


def KS(ser, y, var_type):
    """变量KS, AUC."""
    cut = gen_cut(ser, var_type, n=20, mthd='eqqt', precision=4)
    cross, cut = gen_cross(ser, y, cut, var_type)
    cross.loc[:, 'event_prop'] = cross.loc[:, 1] / cross.sum(axis=1)
    prop_dict = cross.loc[:, 'event_prop'].to_dict()
    ser_prop = apply_woe(ser, cut, prop_dict, var_type)
    ks = calks(y, ser_prop)
    auc = calauc(y, ser_prop)
    return ks, auc


def gen_gaintable(pred, y, bins=20, prob=True, output=False):
    """生成gaintable."""
    t_df = pd.DataFrame({'pred': pred, 'flag': y})
    t_df = t_df.sort_values(by=['pred'], ascending=not(prob))
    t_df['range'] = range(len(t_df))
    t_df['cut'] = pd.cut(t_df['range'], bins)
    t_df['total'] = 1
    total_bad_num = t_df['flag'].sum()
    total_good_num = len(t_df)-total_bad_num
    score_df = t_df.groupby(['cut'])['pred'].agg(['min', 'max'])\
        .rename(columns={'min': 'min_score', 'max': 'max_score'})
    score_df = score_df.applymap(lambda x: round(x, 4))
    score_df['score_range'] = score_df.apply(
            lambda x: pd.Interval(x['min_score'], x['max_score']), axis=1)
    num_df = t_df.groupby(['cut'])['flag'].agg(['sum', 'count'])\
        .rename(columns={'sum': 'bad_num', 'count': 'total'})
    num_df['good_num'] = num_df['total']-num_df['bad_num']
    num_df['bad_rate'] = num_df['bad_num']/num_df['total']
    num_df.sort_index(ascending=True, inplace=True)
    num_df['cum_bad'] = num_df['bad_num'].cumsum()
    num_df['cum_num'] = num_df['total'].cumsum()
    num_df['cum_good'] = num_df['cum_num'] - num_df['cum_bad']
    num_df['cumbad_rate'] = num_df['cum_bad']/num_df['cum_num']
    sample = score_df[['score_range']].merge(num_df, left_index=True,
                                             right_index=True, how='left')
    sample['gain'] = sample['cum_bad']/total_bad_num
    sample['lift'] = sample['gain']/((sample['cum_num'])/(len(t_df)))
    sample['ks'] = np.abs(
            sample['cum_good']/total_good_num
            - sample['cum_bad']/total_bad_num)
    sample[['bad_rate', 'cumbad_rate', 'gain', 'lift', 'ks']] = sample[
            ['bad_rate', 'cumbad_rate', 'gain', 'lift', 'ks']].applymap(
            lambda x: round(x, 4))
    if output:
        sample.to_csv(r'gaintable.csv')
    return sample[['score_range', 'total', 'bad_num', 'bad_rate',
                   'cumbad_rate', 'gain', 'ks', 'lift']]

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:20:38 2021

@author: linjianing
"""


import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from autolrscorecard.utils.validate import param_in_validate
import autolrscorecard.variable_types.variable as vtype


def gen_cut_summ(ser, n=10, mthd='eqqt', precision=4):
    """
    生成连续变量的切分点.

    若序列中只有一个值，返回字符串"N_CUT ERROR"
    input
        ser         序列
        n           切点个数
        mthd        切分方式
            eqqt    等频
            eqdist  等距
    """
    param_in_validate(mthd, ['eqqt', 'eqdist'], '使用了mthd={0}, 仅支持mthd={1}')
    rcut = list(sorted(ser.dropna().unique()))
    if len(rcut) <= 2:
        return [-np.inf, np.inf]
    if mthd == 'eqqt':
        cut = list(np.unique(pd.qcut(ser, n, retbins=True, duplicates='drop')
                             [1].round(precision)))
    elif mthd == 'eqdist':
        cut = list(np.unique(pd.cut(ser, n, retbins=True, duplicates='drop')[1]
                   .round(precision)))
    cut[0] = -np.inf
    cut[-1] = np.inf
    return cut


def gen_cut_count(ser):
    """生成计数变量的切分点."""
    rcut = list(sorted(ser.dropna().unique()))
    if len(rcut) <= 1:
        return [-np.inf, np.inf]
    rcut[0] = -np.inf
    rcut[-1] = np.inf
    return rcut


def gen_cut_discrete(ser):
    """生成分类变量切分点."""
    return {i: val for i, val in enumerate(ser.categories)}


def gen_cut(ser, var_type, **kwargs):
    """生成切分点."""
    if issubclass(var_type, (vtype.Summ, np.number)):
        summ_kw = {key: val for key, val in kwargs.items()
                   if key in ['n', 'mthd', 'precision']}
        return gen_cut_summ(ser, **summ_kw)
    if issubclass(var_type, vtype.Count):
        return gen_cut_count(ser)
    if issubclass(var_type, vtype.Discrete):
        return gen_cut_discrete(ser)


def gen_cross(ser, y, cut, var_type):
    """生成变量的列联表."""
    x = ser.name
    df = pd.DataFrame({x: ser, 'y': y})
    # 切分后返回bin[0, 1, ...]
    if not issubclass(var_type, vtype.Discrete):
        df.loc[:, x] = pd.cut(ser, cut, labels=False, duplicates='drop')
    cross = df.groupby([x, 'y']).size().unstack()
    cross.columns = cross.columns.astype('int')
    allsize = df.groupby([y]).size()
    na_cross = pd.DataFrame({
        0: np.nansum([allsize.loc[0], -cross.sum().loc[0]]),
        1: np.nansum([allsize.loc[1], -cross.sum().loc[1]])},
        index=[-1])
    if issubclass(var_type, vtype.Discrete):
        if not var_type.ordered:
            cross['eventRate'] = cross[1]/np.nansum(cross, axis=1)
            cross.sort_values('eventRate', ascending=False, inplace=True)
            cross.drop(['eventRate'], inplace=True, axis=1)
        cross = cross.loc[cross.sum(axis=1) > 0, :]
        t_cut = {v: k for k, v in enumerate(list(cross.index))}
    else:
        t_cut = [cut[int(x+1)] for x in cross.index]
        t_cut.insert(0, -np.inf)
    cross.reset_index(inplace=True, drop=True)
    cross = cross.append(na_cross)
    cross.fillna(0, inplace=True)
    return cross, t_cut


def apply_woe(ser, cut, woe, var_type):
    """woe应用."""
    if issubclass(var_type, vtype.Discrete):
        return ser.map(cut).map(woe).fillna(woe.get(-1, 0))
    return pd.cut(ser, cut, labels=False).map(woe).fillna(woe.get(-1, 0))


def apply_cut_bin(ser, cut, var_type):
    """Cut to bin."""
    if issubclass(var_type, vtype.Discrete):
        return ser.map(cut).fillna(-1)
    return pd.cut(ser, cut, labels=False).fillna(-1)


def caliv(cross):
    """计算IV值，输入矩阵."""
    warnings.filterwarnings('ignore')
    mat = cross.values
    mat = np.where(mat == 0, 1e-5, mat)
    magc = mat.sum(axis=0)
    entropy = (mat[:, 1]/magc[1] - mat[:, 0]/magc[0])*np.log(
            (mat[:, 1]/magc[1]) / (mat[:, 0]/magc[0]))
    warnings.filterwarnings('default')
    return np.nansum(np.where((entropy == np.inf) | (entropy == -np.inf),
                              0, entropy))


def gen_ksseries(y, pred, pos_label=None):
    """生成KS及KS曲线相应元素.

    input
        y           实际flag
        pred        预测分数
        pos_label   标签被认为是响应，其他作为非响应
    """
    fpr, tpr, thrd = roc_curve(y, pred, pos_label=pos_label)
    eventNum = np.sum(y.to_list())
    allNum = len(y)
    nonEventNum = allNum - eventNum
    # KS对应的x点
    ksTile = (eventNum*tpr + nonEventNum*fpr)/allNum
    return np.max(tpr - fpr), tpr - fpr, ksTile


def calks(y, pred, pos_label=None):
    """计算ks值."""
    return gen_ksseries(y, pred, pos_label)[0]


def calauc(y, pred, pos_label=None):
    """计算auc值."""
    fpr, tpr, thrd = roc_curve(y, pred, pos_label)
    return auc(fpr, tpr)

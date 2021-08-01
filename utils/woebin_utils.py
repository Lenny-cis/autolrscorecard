# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:23:33 2021

@author: linjianing
"""


import sys
import warnings
import numpy as np
import pandas as pd
import scipy.stats as sps
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from functools import reduce
import autolrscorecard.variable_types.variable as vtype


def make_tqdm_iterator(**kwargs):
    """产生tqdm进度条迭代器."""
    options = {
        "file": sys.stdout,
        "leave": True
    }
    options.update(kwargs)
    iterator = tqdm(**options)
    return iterator


def is_shape_I(values):
    """判断输入的列表/序列是否为单调递增."""
    if len(values) < 2:
        return False
    if np.array([values[i] < values[i+1]
                for i in range(len(values)-1)]).all():
        return True
    return False


def is_shape_D(values):
    """判断输入的列表/序列是否为单调递减."""
    if len(values) < 2:
        return False
    if np.array([values[i] > values[i+1]
                for i in range(len(values)-1)]).all():
        return True
    return False


def is_shape_U(values):
    """判断输入的列表/序列是否为先单调递减后单调递增."""
    if len(values) < 3:
        return False
    if not (is_shape_I(values) and is_shape_D(values)):
        knee = np.argmin(values)
        if is_shape_D(values[: knee+1]) and is_shape_I(values[knee:]):
            return True
    return False


def is_shape_A(self, values):
    """判断输入的列表/序列是否为先单调递增后单调递减."""
    if len(values) < 3:
        return False
    if not (is_shape_I(values) and is_shape_D(values)):
        knee = np.argmax(values)
        if is_shape_I(values[: knee+1]) and is_shape_D(values[knee:]):
            return True
    return False


def gen_badrate(arr):
    """输入bin和[0, 1]的列联表，生成badrate."""
    return arr[:, 1]/arr.sum(axis=1)


def arr_badrate_shape(arr):
    """判断badrate的单调性."""
    badRate = gen_badrate(arr)
    if is_shape_I(badRate):
        return 'I'
    elif is_shape_D(badRate):
        return 'D'
    elif is_shape_U(badRate):
        return 'U'
    return np.nan


def slc_min_dist(df):
    """
    选取最小距离.

    计算上下两个bin之间的距离，计算原理参考用惯量类比距离的wald法聚类计算方式
    返回需要被合并的箱号
    """
    R_margin = df.sum(axis=1)
    C_margin = df.sum(axis=0)
    n = df.sum().sum()
    A = df.div(R_margin, axis=0)
    R = R_margin/n
    C = C_margin/n
    # 惯量类比距离
    dist = (A-A.shift()).dropna().applymap(np.square)\
        .div(C, axis=1).sum(axis=1)*(R*R.shift()/(R+R.shift())).dropna()
    return dist.idxmin()


def cut_adj(cut, bin_idxs, variable_type):
    """切分点调整."""
    if not issubclass(variable_type, (vtype.Discrete, pd.CategoricalDtype)):
        return [x for i, x in enumerate(cut) if i not in bin_idxs]
    t_cut = deepcopy(cut)
    for idx in bin_idxs[::-1]:
        t_d = {k: v-1 for k, v in t_cut.items() if v >= idx}
        t_cut.update(t_d)
    return t_cut


def bagging_indices(arr_idxs, idxlist):
    """
    数组索引根据隔板号入袋.

    Parameters
    ----------
    arr_idxs: list
        列表型索引
    idxlist: list
        arr_idxs的隔板号,1 to len(arr_idxs) - 1
    """
    def bagging(x, y):
        if isinstance(x[y], list):
            return x[: y - 1] + [[x[y - 1]] + x[y]] + x[y + 1:]
        return x[: y - 1] + [x[y - 1: y + 1]] + x[y + 1:]
    # 倒序循环需合并的列表，正序会导致表索引改变，合并出错
    return reduce(bagging, [arr_idxs] + sorted(idxlist, reverse=True))


def merge_arr_by_baggedidx(arr, bagged_idxs):
    """
    根据入袋的索引号合并数组.

    Parameters
    ----------
    arr: np.array
        bin和[0, 1]的数组
    bagged_idxs: list
        已入袋的索引，嵌套列表[[], num, [], num]
    """
    def a1(x):
        if isinstance(x, list):
            return arr[x].sum(axis=0)
        return arr[x]

    return np.array(list(map(a1, bagged_idxs)))


def merge_arr_by_idx(arr, idxlist):
    """
    根据隔板号合并分箱，返回合并后的数组.

    不含缺失组
    Parameters
    ----------
    arr: np.array
        bin和[0, 1]的数组
    idxlist: list
        需要被合并的箱的隔板号,1 to len(arr) - 1
    """
    bidxs = bagging_indices(list(range(arr.shape[0])), idxlist)
    return merge_arr_by_baggedidx(arr, bidxs)


def gen_merged_bin(arr, arr_na, merge_idxs, I_min, U_min, variable_shape,
                   tolerance):
    """生成合并结果."""
    # 根据选取的切点合并列联表
    merged_arr = merge_arr_by_idx(arr, merge_idxs)
    shape = arr_badrate_shape(merged_arr, I_min, U_min)
    # badrate的形状符合先验形状的分箱方式保留下来
    if pd.isna(shape) or (shape not in variable_shape):
        return
    detail = calwoe(merged_arr, arr_na)
    woes = detail['WOE']
    tol = cal_min_tol(woes[:-1])
    if tol <= tolerance:
        return
    chi, p, dof, expFreq =\
        sps.chi2_contingency(merged_arr, correction=False)
    var_entropy = sps.entropy(detail['all_num'][:-1])
    var_bin_ = {
        'detail': detail, 'flogp': -np.log(max(p, 1e-5)), 'tolerance': tol,
        'entropy': var_entropy, 'shape': shape, 'bin_cnt': len(merged_arr)
        }
    return var_bin_


def cal_min_tol(arr):
    """计算woe的最小差值."""
    arr_ = arr.copy()
    return np.min(np.abs(arr_[1:] - arr_[:-1]))


def merge_bin_by_idx(crs, idxlist):
    """
    合并分箱，返回合并后的列联表和切点，合并过程中不会改变缺失组，向下合并的方式.

    input
        df          bin和[0, 1]的列联表
        idxlist     需要合并的箱的索引，列表格式
    """
    cross = crs[crs.index != -1].copy(deep=True).values
    cols = crs.columns
    # 倒序循环需合并的列表，正序会导致表索引改变，合并出错
    for idx in idxlist[::-1]:
        cross[idx] = cross[idx-1: idx+1].sum(axis=0)
        cross = np.delete(cross, idx-1, axis=0)
    cross = pd.DataFrame(cross, columns=cols)\
        .append(crs[crs.index == -1])
    return cross


def merge_lowpct_zero(df, cut, variable_type, thrd_PCT=0.05, thrd_n=None,
                      mthd='PCT'):
    """
    合并个数为0和占比过低的箱，不改变缺失组的结果.

    input
        df          bin和[0, 1]的列联表
        cut         原始切点
        thrd_PCT    占比阈值
        mthd        合并方法
            PCT     合并占比过低的箱
            zero    合并个数为0的箱
    """
    cross = df[df.index != -1].copy(deep=True)
    s = 1
    t_cut = cut.copy()
    total = df.sum().sum()
    thrd_n = thrd_n or total * thrd_PCT
    thrd_PCT = thrd_n / total
    while s:
        row_margin = cross.sum(axis=1)
        min_num = row_margin.min()
        # 找到占比最低的组或个数为0的组
        if mthd.upper() == 'PCT':
            min_idx = row_margin.idxmin()
        else:
            zero_idxs = cross[(cross == 0).any(axis=1)].index
            if len(zero_idxs) >= 1:
                min_idx = zero_idxs[0]
                min_num = 0
            else:
                min_num = np.inf
        # 占比低于阈值则合并
        if min_num/total <= thrd_PCT and cross.shape[0] > 1:
            idxs = list(cross.index)
            # 最低占比的组的索引作为需要合并的组
            # sup_idx确定合并索引的上界，上界不超过箱数
            # inf_idx确定合并索引的下界，下界不低于0
            min_idx_row = idxs.index(min_idx)
            sup_idx = idxs[min(len(cross)-1, min_idx_row+1)]
            inf_idx = idxs[max(0, min_idx_row-1)]
            # 需合并组为第一组，向下合并
            if min_idx == idxs[0]:
                merge_idx = idxs[1]
            # 需合并组为最后一组，向上合并
            elif min_idx == idxs[-1]:
                merge_idx = min_idx
            elif sup_idx == inf_idx:
                merge_idx = inf_idx
            # 介于第一组和最后一组之间，找向上或向下最近的组合并
            else:
                merge_idx = slc_min_dist(cross.loc[inf_idx: sup_idx])
            cross = merge_bin_by_idx(cross, [merge_idx])
            t_cut = cut_adj(t_cut, [merge_idx], variable_type)
        else:
            s = 0
    return cross.append(df[df.index == -1]), t_cut


def calwoe(arr, arr_na, modify=True):
    """计算WOE、IV及分箱细节."""
    warnings.filterwarnings('ignore')
    arr = np.append(arr, arr_na, axis=0)
    col_margin = arr.sum(axis=0)
    row_margin = arr.sum(axis=1)
    event_rate = arr[:, 1] / row_margin
    event_prop = arr[:, 1] / col_margin[1]
    non_event_prop = arr[:, 0] / col_margin[0]
    # 将0替换为极小值，便于计算，计算后将rate为0的组赋值为其他组的最小值，
    # rate为1的组赋值为其他组的最大值
    WOE = np.log(np.where(event_prop == 0, 1e-5, event_prop)
                 / np.where(non_event_prop == 0, 1e-5, non_event_prop))
    WOE[event_rate == 0] = np.min(WOE[:-1][event_rate[:-1] != 0])
    WOE[event_rate == 1] = np.max(WOE[:-1][event_rate[:-1] != 1])
    # 调整缺失组的WOE
    if modify is True:
        if WOE[-1] == max(WOE):
            WOE[-1] = max(WOE[:-1])
        elif WOE[-1] == min(WOE):
            WOE[-1] = 0
    iv = (event_prop - non_event_prop) * WOE
    warnings.filterwarnings('default')
    return {'all_num': row_margin, 'event_rate': event_rate, 'IV': iv,
            'event_num': arr[:, 1], 'WOE': WOE.round(4), 'SUMIV': iv.sum()}


# def calwoe(df, modify=True):
#     """计算WOE、IV及分箱细节."""
#     warnings.filterwarnings('ignore')
#     cross = df.values
#     col_margin = cross.sum(axis=0)
#     row_margin = cross.sum(axis=1)
#     event_rate = cross[:, 1] / row_margin
#     event_prop = cross[:, 1] / col_margin[1]
#     non_event_prop = cross[:, 0] / col_margin[0]
#     # 将0替换为极小值，便于计算，计算后将rate为0的组赋值为其他组的最小值，
#     # rate为1的组赋值为其他组的最大值
#     WOE = np.log(np.where(event_prop == 0, 1e-5, event_prop)
#                  / np.where(non_event_prop == 0, 1e-5, non_event_prop))
#     WOE[event_rate == 0] = np.min(WOE[(event_rate != 0) & (df.index != -1)])
#     WOE[event_rate == 1] = np.max(WOE[(event_rate != 1) & (df.index != -1)])
#     # 调整缺失组的WOE
#     if modify is True:
#         if WOE[df.index == -1] == max(WOE):
#             WOE[df.index == -1] = max(WOE[df.index != -1])
#         elif WOE[df.index == -1] == min(WOE):
#             WOE[df.index == -1] = 0
#     iv = (event_prop-non_event_prop)*WOE
#     warnings.filterwarnings('default')
#     return pd.DataFrame({'all_num': row_margin, 'event_rate': event_rate,
#                          'event_num': cross[:, 1], 'WOE': WOE.round(4), 'IV': iv},
#                         index=df.index).to_dict(orient='index'), iv.sum()



def cut_to_interval(cut, variable_type):
    """切分点转换为字符串区间."""
    bin_cnt = len(cut) - 1
    if not issubclass(variable_type, (vtype.Discrete, pd.CategoricalDtype)):
        cut_str = {int(x): '(' + ','.join([str(cut[x]), str(cut[x+1])]) + ']'
                   for x in range(int(bin_cnt))}
    else:
        d = defaultdict(list)
        for k, v in cut.items():
            d[v].append(str(k))
        cut_str = {int(k): '['+','.join(v)+']' for k, v in d.items()}
    return cut_str

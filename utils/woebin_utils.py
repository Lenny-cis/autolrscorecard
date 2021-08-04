# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:23:33 2021

@author: linjianing
"""


import ray
import warnings
import numpy as np
import pandas as pd
import scipy.stats as sps
from copy import deepcopy
from collections import defaultdict
from itertools import chain
from functools import reduce
import autolrscorecard.variable_types.variable as vtype
from .progress_bar import ProgressBar, make_tqdm_iterator
from time import sleep


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


def slc_min_dist(arr):
    """
    选取最小距离.

    计算上下两个bin之间的距离，计算原理参考用惯量类比距离的wald法聚类计算方式
    返回需要被合并的箱号
    """
    R_margin = arr.sum(axis=1, keepdims=True)
    C_margin = arr.sum(axis=0, keepdims=True)
    n = arr.sum()
    A = np.divide(arr, R_margin)
    R = R_margin/n
    C = C_margin/n
    # 惯量类比距离
    dist = np.divide(np.square(A[1:]-A[:-1]), C).sum(axis=1, keepdims=True)\
        / (1/R[1:]+1/R[:-1])
    return dist.argmin() + 1


def cut_adjust(cut, bin_idxs):
    """
    切分点调整.

    Parameters
    ----------
    cut: list or dict
        连续型为list的cut
        分类型为dict的cut
    bin_idxs: list
        隔板号,1 to len(cut) - 1 or len(cut.keys()) - 1
    """
    def cut_dict_adjust(cut_dict, idx):
        return {k: v-1 if v >= idx else v for k, v in cut_dict.items()}

    if isinstance(cut, list):
        return [x for i, x in enumerate(cut) if i not in bin_idxs]
    return reduce(cut_dict_adjust, [cut] + sorted(bin_idxs, reverse=True))


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
            return x[: y-1] + [[x[y-1]] + x[y]] + x[y+1:]
        return x[: y-1] + [x[y-1: y+1]] + x[y+1:]
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
    shape = arr_badrate_shape(merged_arr)
    # badrate的形状符合先验形状的分箱方式保留下来
    if pd.isna(shape) or (shape not in variable_shape):
        return
    elif shape in ['I', 'D']:
        if merged_arr.shape[0] < I_min:
            return
    else:
        if merged_arr.shape[0] < U_min:
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


def best_merge_by_idx(arr, idx):
    """."""
    # inf_idx确定合并索引的下界，下界不低于1
    if idx == 0:
        merge_idx = 1
    # sup_idx确定合并索引的上界，上界不超过箱数
    elif idx == arr.shape[0] - 1:
        merge_idx = idx
    else:
        merge_idx = slc_min_dist(arr[idx-1: idx+2, :]) + idx - 1
    return merge_arr_by_idx(arr, [merge_idx]), merge_idx


def merge_lowpct(arr, thrd_PCT):
    """
    合并占比过低的箱.

    Parameters
    ----------
    arr
        bin和[0, 1]的数组
    thrd_PCT
        占比阈值

    Return
    ----------
    arr: np.array
    idxlist: list
    """
    arr = arr.copy()
    total = arr.sum()
    arr_idxs = list(range(arr.shape[0]))
    idxlist = []
    while True:
        row_margin = arr.sum(axis=1)
        min_idx = row_margin.argmin()
        min_num = row_margin[min_idx]
        len_arr = arr.shape[0]
        # 占比低于阈值则合并
        if min_num/total <= thrd_PCT and len_arr > 1:
            arr, merge_idx = best_merge_by_idx(arr, min_idx)
            bulkhead = arr_idxs.pop(merge_idx)
            idxlist.append(bulkhead)
        else:
            return arr, sorted(idxlist)


def merge_fewnum(arr, thrd_n):
    """
    合并个数少的箱.

    Parameters
    ----------
    arr
        bin和[0, 1]的数组
    thrd_n
        箱中总数的最少样本数

    Return
    ----------
    arr: np.array
    idxlist: list
    """
    thrd_pct = thrd_n / arr.sum()
    return merge_lowpct(arr, thrd_pct)


def merge_zeronum(arr):
    """
    合并个数为0箱.

    Parameters
    ----------
    arr
        bin和[0, 1]的数组

    Return
    ----------
    arr: np.array
    idxlist: list
    """
    arr = arr.copy()
    arr_idxs = list(range(arr.shape[0]))
    zero_ = np.array(arr_idxs)[(arr == 0).any(axis=1)]
    idxlist = []
    while zero_.shape[0] > 0:
        # 占比低于阈值则合并
        min_idx = zero_[0]
        zero_ = zero_[1:]
        arr, merge_idx = best_merge_by_idx(arr, min_idx)
        bulkhead = arr_idxs.pop(merge_idx)
        idxlist.append(bulkhead)
    return arr, sorted(idxlist)


def merge_lowpct_zero(arr, cut, thrd_PCT=0.03, thrd_n=None):
    """
    合并地占比和0的箱.

    Parameters
    ----------
    arr : np.array
        bin和[0, 1]的数组.
    cut: list
        切点集合
    thrd_PCT : float
        占比阈值.
    thrd_n : int, optional
        箱中总数的最少样本数. The default is None. 如果提供则仅考虑数量不考虑占比.

    Returns
    -------
    arr: np.array
        合并的结果
    cut: list
        合并调整后的cut
    """
    arr, idxlist = merge_zeronum(arr)
    cut = cut_adjust(cut, idxlist)
    if thrd_n is not None:
        arr, idxlist = merge_fewnum(arr, thrd_n)
        cut = cut_adjust(cut, idxlist)
        return arr, cut
    arr, idxlist = merge_lowpct(arr, thrd_PCT)
    cut = cut_adjust(cut, idxlist)
    return arr, cut


def calwoe(arr, arr_na, precision=4, modify=True):
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
            'event_num': arr[:, 1], 'WOE': WOE.round(precision),
            'SUMIV': iv.sum()}


def cut_diff_ptp(cut, qt):
    """
    cut前后两个值之间的差的max-min.

    Parameters
    ----------
    cut : list
        切分点，[-np.inf, ..., np.inf].
    qt : list
        四分位列表[min, q25, q50, q75, max].

    Returns
    -------
    float or int
        切点差值的差值.

    """
    clip_min, clip_max = np.clip(
        qt,
        2.5 * qt[1] - 1.5 * qt[3], 2.5 * qt[3] - 1.5 * qt[1])[[0, -1]]
    x = deepcopy(cut)
    x[0] = qt[0] if clip_min > x[1] else clip_min
    x[-1] = qt[-1] if clip_max < x[-2] else clip_max
    return np.ptp(np.diff(x))


def normalize(x):
    """
    归一化.

    Parameters
    ----------
    x : series or array

    Returns
    -------
    series or array

    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min)/(x_max - x_min)


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


@ray.remote
def _gen_var_bin(arr, arr_na, bc, I_min, U_min, variable_shape,
                 tolerance, cut, qt, pba):
    """使用ray封装合并箱体的函数."""
    var_bin = [gen_var_bin(arr, arr_na, merge_idxs, I_min, U_min,
                           variable_shape, tolerance, cut, qt)
               for merge_idxs in bc]
    pba.update.remote(len(var_bin))
    return var_bin


def parallel_gen_var_bin(bcs, arr, arr_na, I_min, U_min,
                         variable_shape, tolerance, cut, qt, desc, lbcs):
    """使用RAY多核计算."""
    pb = ProgressBar(lbcs, desc)
    actor = pb.actor
    refs = [_gen_var_bin.remote(
        arr, arr_na, bc, I_min, U_min, variable_shape, tolerance, cut,
        qt, actor)
        for bc in bcs]
    pb.print_until_done()
    remote_var_bins = ray.get(refs)
    var_bins = list(chain.from_iterable(remote_var_bins))
    return var_bins


def gen_var_bin(arr, arr_na, merge_idxs, I_min, U_min, variable_shape,
                tolerance, cut, qt):
    """计算变量分箱结果."""
    var_bin = gen_merged_bin(arr, arr_na, merge_idxs, I_min, U_min,
                             variable_shape, tolerance)
    cut = cut_adjust(cut, merge_idxs)
    mindiffstep = cut_diff_ptp(cut, qt)
    if var_bin is not None:
        var_bin.update({'cut': cut, 'mindiffstep': mindiffstep})
    return var_bin


def one_core_gen_var_bin(bcs, arr, arr_na, I_min, U_min,
                         variable_shape, tolerance, cut, qt, desc, lbcs):
    """使用单核计算."""
    def yield_var_bin():
        for bc in bcs:
            var_bin = [gen_var_bin(arr, arr_na, merge_idxs, I_min, U_min,
                                   variable_shape, tolerance, cut, qt)
                       for merge_idxs in bc]
            yield var_bin

    tqdm_options = {'total': lbcs, 'desc': desc, 'disable': False}
    var_bins = []
    with make_tqdm_iterator(**tqdm_options) as progressbar:
        for bc in bcs:
            print(bc)
            vbrest = [gen_var_bin(arr, arr_na, merge_idxs, I_min, U_min,
                                  variable_shape, tolerance, cut, qt)
                      for merge_idxs in bc]
            var_bins.extend(vbrest)
            progressbar.update(len(vbrest))
    return var_bins

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

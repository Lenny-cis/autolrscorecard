# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 23:24:23 2021

@author: Lenny
"""

import ray
import numpy as np
from autolrscorecard.utils import (
    normalize, gen_merged_bin, cut_adjust, cut_diff_ptp,
    ProgressBar, make_tqdm_iterator)


def tiny_work(x):
    # time.sleep(0.0001) # replace this is with work you need to do
    return x


@ray.remote
def mega_work(start, end, pba):
    a = [tiny_work(x) for x in range(start, end)]
    pba.update.remote(1)
    return a


def rr(xx):
    num_ticks = xx
    pb = ProgressBar(num_ticks, 'tt')
    actor = pb.actor
    # [result_ids.append(mega_work.remote(x*1000, (x+1)*1000)) for x in range(100)]
    result_ids = [mega_work.remote(x*1000, (x+1)*1000, actor) for x in range(xx)]
    pb.print_until_done()
    results = ray.get(result_ids)
    return results


def test_b(arr1):
    return {'arr': arr1}


@ray.remote
def _gen_var_bin(arr, arr_na, merge_idxs, I_min, U_min, variable_shape,
                 tolerance, cut, qt, pba):
    """使用ray封装合并箱体的函数."""
    var_bin = test_b(np.array([1, 2, 3, 4, 5, 6]))
    # var_bin = {'a': normalize(np.array([1, 2, 3, 4, 5, 6]))}
    # var_bin = gen_var_bin(arr, arr_na, merge_idxs, I_min, U_min,
    #                       variable_shape, tolerance, cut, qt)
    pba.update.remote(1)
    return var_bin


def parallel_gen_var_bin(bcs, arr, arr_na, I_min, U_min,
                         variable_shape, tolerance, cut, qt, desc, lbcs):
    """使用RAY多核计算."""
    pb = ProgressBar(lbcs, desc)
    actor = pb.actor
    refs = [_gen_var_bin.remote(
        arr, arr_na, merge_idxs, I_min, U_min, variable_shape, tolerance, cut,
        qt, actor)
        for merge_idxs in bcs]
    pb.print_until_done()
    var_bins = ray.get(refs)
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
        for merge_idxs in bcs:
            var_bin = gen_var_bin(arr, arr_na, merge_idxs, I_min, U_min,
                                  variable_shape, tolerance, cut, qt)
            yield var_bin

    tqdm_options = {'total': lbcs, 'desc': desc}
    var_bins = []
    with make_tqdm_iterator(**tqdm_options) as progress_bar:
        for vbrest in yield_var_bin():
            var_bins.append(vbrest)
            progress_bar.update()
    return var_bins

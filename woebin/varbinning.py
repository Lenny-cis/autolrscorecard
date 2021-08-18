# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 16:39:10 2021

@author: linjianing
"""


import pandas as pd
import numpy as np
from copy import deepcopy
from itertools import (product, combinations)
import autolrscorecard.variable_types.variable as vtype
from autolrscorecard.utils.validate import (
    param_in_validate, param_contain_validate)
from autolrscorecard.utils import (
    gen_cut, gen_cross, apply_woe, apply_cut_bin, merge_lowpct_zero,
    make_tqdm_iterator, parallel_gen_var_bin,
    woe_list2dict, gen_var_bin)
from autolrscorecard.plotfig import plot_bins_set


class VarBinning:
    """探索性分箱."""

    def __init__(self, cut_cnt=50, thrd_PCT=0.025, thrd_n=None,
                 max_bin_cnt=6, I_min=3, U_min=4, cut_mthd='eqqt',
                 variable_shape='IDU', tolerance=0,
                 variable_type=vtype.Summ(2), n_jobs=-1, **kwargs):
        self.cut_cnt = cut_cnt
        self.thrd_PCT = thrd_PCT
        self.thrd_n = thrd_n
        self.max_bin_cnt = max_bin_cnt
        self.I_min = I_min
        self.U_min = U_min
        self.cut_mthd = cut_mthd
        self.variable_shape = list(variable_shape)
        self.variable_type = type(variable_type)
        self.prec = variable_type.prec
        self.tolerance = tolerance
        self.n_jobs = n_jobs
        self.describe = kwargs.get('describe', '未知')
        self.data = {'bins_set': {}, 'best_bins': {}, 'selected_best': {}}
        param_in_validate(
            self.cut_mthd, ['eqqt', 'eqdist'],
            '使用了cut_mthd={0}, 仅支持cut_mthd={1}')
        param_contain_validate(
            self.variable_shape, ['I', 'D', 'U', 'A'],
            '使用了variable_shape={0}, 仅支持variable_shape={1}')

    @property
    def bins_set(self):
        """分箱集合."""
        return self.data['bins_set']

    @bins_set.setter
    def bins_set(self, bs):
        """分箱集合."""
        self.data['bins_set'] = bs

    @property
    def best_bins(self):
        """最优分箱."""
        return self.data['best_bins']

    @best_bins.setter
    def best_bins(self, bb):
        """最优分箱."""
        self.data['best_bins'] = bb

    @property
    def selected_best(self):
        """已选择的最优分箱."""
        return self.data['selected_best']

    @selected_best.setter
    def selected_best(self, bb):
        """已选择的最优分箱."""
        self.data['selected_best'] = bb

    def _gen_comb_bins(self, crs, crs_na, cut):
        def comb_comb(hulkheads, loops):
            for loop in loops:
                yield combinations(hulkheads, loop)

        cross = crs.copy()
        minnum_bin_I = self.I_min
        minnum_bin_U = self.U_min
        vs = self.variable_shape
        maxnum_bin = self.max_bin_cnt
        qt = self.quantile
        tol = self.tolerance
        describe = self.indep
        if 'U' not in vs:
            minnum_bin = minnum_bin_I
        elif 'I' not in vs and 'D' not in vs:
            minnum_bin = minnum_bin_U
        else:
            minnum_bin = min(minnum_bin_I, minnum_bin_U)
        rawnum_bin = cross.shape[0]
        hulkhead_list = list(range(1, cross.shape[0]))
        # 限定分组数的上下限
        maxnum_hulkhead_loops = max(rawnum_bin - minnum_bin, 0)
        minnum_hulkhead_loops = max(rawnum_bin - maxnum_bin, 0)
        loops_ = range(minnum_hulkhead_loops, maxnum_hulkhead_loops)
        bcs = comb_comb(hulkhead_list, loops_)
        # 多核并行计算
        var_bins = parallel_gen_var_bin(
            bcs, crs, crs_na, minnum_bin_I, minnum_bin_U, vs, tol, cut,
            qt, describe, self.n_jobs)
        var_bin_dic = {k: v for k, v in enumerate(var_bins) if v is not None}
        return var_bin_dic

    def fit(self, x, y):
        """单变量训练."""
        if x.dropna().nunique() <= 1:
            return self
        self.dep = y.name
        self.indep = x.name
        if issubclass(self.variable_type, (vtype.Summ, vtype.Count)):
            self.quantile = np.percentile(
                x.dropna(), [0, 25, 50, 75, 100]).tolist()
        else:
            self.quantile = []
        cut = gen_cut(x, self.variable_type,
                      n=self.cut_cnt, mthd=self.cut_mthd,
                      precision=self.prec)
        cross, cut = gen_cross(x, y, cut, self.variable_type)
        if (cross.loc[cross.index != -1, 0] == 0).all()\
                or (cross.loc[cross.index != -1, 1] == 0).all():
            return self
        crs = cross.loc[cross.index != -1, :].values
        crs_na = cross.loc[cross.index == -1, :].values
        crs, cut = merge_lowpct_zero(crs, cut, self.thrd_PCT, self.thrd_n)
        bin_dic = self._gen_comb_bins(crs, crs_na, cut)
        self.bins_set = bin_dic
        return self

    def manual_fit(self, x, y, subj_cut):
        """手工分箱."""
        if x.dropna().nunique() <= 1:
            return self
        if issubclass(self.variable_type, (vtype.Summ, vtype.Count)):
            quantile = np.percentile(
                x.dropna(), [0, 25, 50, 75, 100]).tolist()
        else:
            quantile = []
        cut = deepcopy(subj_cut)
        cross, cut = gen_cross(x, y, cut, self.variable_type)
        if (cross.loc[cross.index != -1, 0] == 0).all()\
                or (cross.loc[cross.index != -1, 1] == 0).all():
            return self
        crs = cross.loc[cross.index != -1, :].values
        crs_na = cross.loc[cross.index == -1, :].values
        crs, cut = merge_lowpct_zero(crs, cut, self.thrd_PCT, self.thrd_n)
        bins_set = gen_var_bin(crs, crs_na, [], 0, 0, 'IDUA', 0, cut,
                               quantile)
        self.selected_best = bins_set
        return self

    def fast_search_best(self, slc_mthd='IV', tolerance=0.1):
        """单一目标搜索."""
        if slc_mthd != 'IV':
            slc_mthd = [slc_mthd, 'IV']
        else:
            slc_mthd = ['IV']
        bins_set = self.cal_sort_values()
        if len(bins_set) == 0:
            return self
        bins = pd.DataFrame.from_dict(bins_set, orient='index')
        filter_bins = bins.loc[bins.loc[:, 'tolerance'] > tolerance, :]
        filter_res = filter_bins.sort_values(by=slc_mthd, ascending=False)\
            .iloc[0, :].to_dict()
        self.selected_best = filter_res
        return self

    def cal_sort_values(self):
        """计算排序的指标值."""
        def _normalize(x):
            x_min = x.min()
            x_max = x.max()
            return (x - x_min)/(x_max - x_min)

        bins_set = self.bins_set
        if len(bins_set) == 0:
            return {}
        bins = pd.DataFrame.from_dict(bins_set, orient='index')
        norm_iv = _normalize(bins.loc[:, 'IV'])
        norm_e = _normalize(bins.loc[:, 'entropy'])
        bins.loc[:, 'ivae'] = np.hypot(norm_iv, norm_e)
        return bins.to_dict(orient='index')

    def transform(self, x, retbin=False):
        """单变量应用."""
        best_ = self.selected_best
        cut = best_['cut']
        woe = woe_list2dict(best_['detail']['WOE'])
        name = str(x.name)
        if not retbin:
            trns = apply_woe(x, cut, woe, self.variable_type)
        else:
            trns = apply_cut_bin(x, cut, self.variable_type)
        trns.name = name
        return trns

    def grid_search_best(
            self, verbose=True, bins_cnt=None, variable_shape=None,
            slc_mthd=['ivae', 'flogp', 'entropy', 'IV', 'mindiffstep'],
            tolerance=np.linspace(0.0, 0.1, 11).tolist(), n_jobs=-1, **kwargs):
        """网格选择最优分箱."""
        def _deft_null(x, dx):
            if x is None:
                return dx
            if isinstance(x, list):
                return x
            return [x]

        bins_cnt = _deft_null(
            bins_cnt,
            list(range(min(self.I_min, self.U_min), self.max_bin_cnt)))
        slc_mthd = _deft_null(
            slc_mthd, ['ivae', 'flogp', 'entropy', 'IV', 'mindiffstep'])
        tolerance = [tolx for tolx in tolerance if tolx >= self.tolerance]
        variable_shape = variable_shape or self.variable_shape
        param_contain_validate(
            slc_mthd, ['flogp', 'IV', 'entropy', 'ivae', 'mindiffstep'],
            '使用了slc_mthd={0}, 仅支持slc_mthd={1}')
        bins_set = self.cal_sort_values()
        if len(bins_set) == 0:
            return self
        bins = pd.DataFrame.from_dict(bins_set, orient='index')\
            .loc[:, slc_mthd + ['shape', 'tolerance', 'bin_cnt']].to_records()
        prod_ = list(product(bins_cnt, variable_shape, slc_mthd, tolerance))
        if len(prod_) <= 0:
            return self
        tqdm_options = {'iterable': prod_, 'desc': self.indep}
        progress_bar = make_tqdm_iterator(**tqdm_options)
        exists = {}
        best_bins = {}
        for prod in progress_bar:
            cnt, shape, mthd, tol = prod
            filter_bins = bins[
                (bins['bin_cnt'] == cnt) & (bins['shape'] == shape)
                & (bins['tolerance'] >= tol)][[mthd, 'index']]
            if len(filter_bins) <= 0:
                continue
            parm = (('bins_cnt', cnt), ('variable_shape', shape),
                    ('slc_mthd', mthd), ('tolerance', tol))
            filter_bins.sort(order=mthd)
            tem_dic_key = filter_bins['index'][-1]
            if mthd == 'mindiffstep':
                tem_dic_key = filter_bins['index'][0]
            if tem_dic_key in exists.keys():
                del best_bins[exists[tem_dic_key]]
            exists.update({tem_dic_key: parm})
            best_bins.update({parm: bins_set[tem_dic_key]})
        self.best_bins = best_bins
        return self

    def plot_selected(self, save_path=None):
        """画出被选中的最优分箱."""
        if len(self.selected_best) <= 0:
            return self
        plot_bins_set(self.selected_best, self.variable_type, self.indep,
                      save_path)

    def plot_best_bins(self, save_path=None):
        """画出所有搜索到的最优分箱."""
        if len(self.best_bins) <= 0:
            return self
        plot_bins_set(self.best_bins, self.variable_type, self.indep,
                      save_path)
        return self

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 16:39:10 2021

@author: linjianing
"""


import pandas as pd
import numpy as np
import itertools as its
import scipy.stats as sps
from copy import deepcopy
from itertools import product
import variable_types.variable as vtype
from utils.validate import param_in_validate, param_contain_validate
from utils.performance_utils import gen_cut, gen_cross, apply_woe, apply_cut_bin
from utils.woebin_utils import (merge_lowpct_zero, make_tqdm_iterator,
                                merge_bin_by_idx, bad_rate_shape, cut_adj,
                                calwoe, cut_to_interval)
from plotfig.plotfig import plot_bin


PBAR_FORMAT = "Possible: {total} | Elapsed: {elapsed} | Progress: {l_bar}{bar}"


class VarBinning:
    """单个变量分箱.

    Input_
        variable_shape: IUDA
        slc_mthd: IV best entropy p
        subjective_cut: [list]

    Return_
        bin_dic: {_id: {detail: {},
                        IV: iv,
                        WOE: {},
                        bin_cnt: cnt,
                        cut: list,
                        describle: str,
                        entropy: entropy,
                        flogp: -log(p),
                        num_inflection: -n_inflection,
                        shape: U/I/D}}}}
        best_dic: upon bin_dic
    """

    def __init__(self, cut_cnt=20, thrd_PCT=0.05, thrd_n=None, verbose=True,
                 max_bin_cnt=6, I_min=3, U_min=4, cut_mthd='eqqt',
                 variable_shape=None, slc_mthd='IV', tolerance=0.1, subjective_cut=None, **kwargs):
        self.cut_cnt = cut_cnt
        self.thrd_PCT = thrd_PCT
        self.thrd_n = thrd_n
        self.max_bin_cnt = max_bin_cnt
        self.max_bin_cnt = max_bin_cnt
        self.I_min = I_min
        self.U_min = U_min
        self.cut_mthd = cut_mthd
        self.variable_shape = list(variable_shape)
        self.slc_mthd = slc_mthd
        self.variable_type = type(kwargs.get('variable_type'))
        self.tolerance = tolerance
        self.describe = kwargs.get('describe', '未知')
        self.verbose = verbose
        self.subjective_cut = subjective_cut
        self.best_dic = {}
        self.bin_dic = {}
        param_in_validate(self.slc_mthd, ['flogp', 'IV', 'entropy', 'best', 'minptp'], '使用了slc_mthd={0}, 仅支持slc_mthd={1}')
        param_in_validate(self.cut_mthd, ['eqqt', 'eqdist'], '使用了cut_mthd={0}, 仅支持cut_mthd={1}')
        param_contain_validate(self.variable_shape, ['I', 'D', 'U', 'A'], '使用了variable_shape={0}, 仅支持variable_shape={1}')

    def _lowpct_zero_merge(self, crs, cut):
        cross = crs.copy(deep=True)
        cross, pct_cut = merge_lowpct_zero(cross, cut, self.variable_type,
                                           thrd_PCT=self.thrd_PCT,
                                           thrd_n=self.thrd_n,
                                           mthd='PCT')
        cross, t_cut = merge_lowpct_zero(cross, pct_cut, self.variable_type,
                                         thrd_PCT=self.thrd_PCT,
                                         thrd_n=self.thrd_n,
                                         mthd='zero')
        return cross, t_cut

    def _gen_comb_bins(self, crs, cut, verbose=False):
        cross = crs.copy(deep=True)
        min_I = self.I_min - 1
        min_U = self.U_min - 1
        if 'U' not in self.variable_shape:
            min_cut_cnt = min_I
        elif 'I' not in self.variable_shape and 'D' not in self.variable_shape:
            min_cut_cnt = min_U
        else:
            min_cut_cnt = min(min_I, min_U)
        max_cut_cnt = self.max_bin_cnt - 1
        cut_point_list = [x for x in cross.index[:] if x != -1][1:]
        cut_point_cnt = len(cut_point_list)
        # 限定分组数的上下限
        max_cut_loops_cnt = cut_point_cnt - min_cut_cnt + 1
        min_cut_loops_cnt = max(cut_point_cnt - max_cut_cnt, 0)
        var_bin_dic = {}
        s = 0
        loops_ = range(min_cut_loops_cnt, max_cut_loops_cnt)
        bcs = [bi for loop in loops_ for bi in its.combinations(cut_point_list, loop)]
        if len(bcs) > 0:
            tqdm_options = {'bar_format': PBAR_FORMAT,
                            'total': len(bcs),
                            'desc': self.indep,
                            'disable': True}
            if verbose:
                tqdm_options.update({'disable': False})
            with make_tqdm_iterator(**tqdm_options) as progress_bar:
                for bin_idxs in bcs:
                    # 根据选取的切点合并列联表
                    merged = merge_bin_by_idx(cross, bin_idxs)
                    shape = bad_rate_shape(merged, self.I_min, self.U_min)
                    # badrate的形状符合先验形状的分箱方式保留下来
                    if pd.isna(shape) or shape not in self.variable_shape:
                        progress_bar.update()
                        continue
                    detail, iv = calwoe(merged)
                    woe = {key: val['WOE'] for key, val in detail.items() if key != -1}
                    woe_ = pd.Series(woe)
                    tol = np.min(np.abs(woe_ - woe_.shift()))
                    if tol <= self.tolerance:
                        progress_bar.update()
                        continue
                    var_bin_dic[s] = {}
                    chi, p, dof, expFreq =\
                        sps.chi2_contingency(
                                merged.loc[~merged.index.isin([-1]), :].values,
                                correction=False)
                    var_entropy = sps.entropy(pd.Series(
                        [val['all_num'] for key, val in detail.items()
                         if key != -1]))
                    var_bin_dic[s] = {
                        'detail': detail,
                        'IV': iv,
                        'flogp': -np.log(max(p, 1e-5)),
                        'entropy': var_entropy,
                        'shape': shape,
                        'bin_cnt': len(merged)-1,
                        'cut': cut_adj(cut, bin_idxs, self.variable_type),
                        'WOE': woe,
                        'describe': self.describe,
                        'quantile': self.quantile}
                    s += 1
                    progress_bar.update()
        return var_bin_dic

    def _select_best(self, dic):
        def _normalize(x):
            x_min = x.min()
            x_max = x.max()
            return (x - x_min)/(x_max - x_min)

        def _ptp(x, qt):
            if isinstance(x, dict):
                return -1 * np.ptp(x.values.value_counts())
            clip_min, clip_max = np.clip(qt, 2.5 * qt[1] - 1.5 * qt[3], 2.5 * qt[3] - 1.5 * qt[1])[[0, -1]]
            x = deepcopy(x)
            x[0] = qt[0] if clip_min > x[1] else clip_min
            x[-1] = qt[-1] if clip_max < x[-2] else clip_max
            # print(qt, [clip_min, clip_max], x, np.diff(x))
            return -1 * np.ptp(np.diff(x))

        if len(dic) == 0:
            return {}
        dd = pd.DataFrame.from_dict(dic, orient='index')
        dd.loc[:, 'num_inflection'] = dd.loc[:, 'shape'].map({'I': 0, 'D': 0, 'U': -1})
        dd.loc[:, 'reverse_ptp'] = dd.apply(lambda x: _ptp(x['cut'], x['quantile']), axis=1)
        if self.slc_mthd == 'flogp':
            sort_keys = ['num_inflection', 'flogp', 'bin_cnt', 'entropy', 'IV']
        elif self.slc_mthd == 'IV':
            sort_keys = ['num_inflection', 'IV', 'bin_cnt', 'entropy', 'flogp']
        elif self.slc_mthd == 'entropy':
            sort_keys = ['num_inflection', 'entropy', 'bin_cnt', 'IV', 'flogp']
        elif self.slc_mthd == 'minptp':
            sort_keys = ['num_inflection', 'reverse_ptp', 'bin_cnt', 'IV', 'flogp', 'entropy']
        elif self.slc_mthd == 'best':
            norm_iv = _normalize(dd.loc[:, 'IV'])
            norm_e = _normalize(dd.loc[:, 'entropy'])
            dd.loc[:, 'ivae'] = np.hypot(norm_iv, norm_e)
            sort_keys = ['num_inflection', 'ivae', 'bin_cnt']
        best_dd = dd.sort_values(by=sort_keys, ascending=False).iloc[[0], :]
        # best_dd = dd.sort_values(by=sort_keys, ascending=False).groupby(['shape', 'bin_cnt']).head(1)
        return best_dd.to_dict(orient='index')

    def fit(self, x, y):
        """单变量训练."""
        verbose = self.verbose
        if x.dropna().nunique() <= 1:
            # if verbose:
            #     print('nunique <= 1: {}, {}'.format(x.name, list(x.unique())))
            return self
        self.dep = y.name
        self.indep = x.name
        if issubclass(self.variable_type, (vtype.Summ, vtype.Count)):
            self.quantile = np.percentile(x.dropna(), [0, 25, 50, 75, 100]).tolist()
        else:
            self.quantile = []
        if self.subjective_cut is not None:
            print('Subjective cut: {}'.format(self.subjective_cut))
            cut = self.subjective_cut
            cross, t_cut = gen_cross(x, y, cut, self.variable_type)
            detail, iv = calwoe(cross)
            woe = {key: val['WOE'] for key, val in detail.items() if key != -1}
            chi, p, dof, expFreq =\
                sps.chi2_contingency(
                        cross.loc[~cross.index.isin([-1]), :].values,
                        correction=False)
            var_entropy = sps.entropy(pd.Series(
                [val['all_num'] for key, val in detail.items()
                 if key != -1]))
            shape = bad_rate_shape(cross, 0, 0)
            self.bin_dic = {0: {
                'detail': detail,
                'IV': iv,
                'flogp': -np.log(max(p, 1e-5)),
                'entropy': var_entropy,
                'shape': shape,
                'bin_cnt': len(cross)-1,
                'cut': cut,
                'WOE': woe,
                'describe': self.describe,
                'quantile': self.quantile}
                }
            self.best_dic = deepcopy(self.bin_dic)
        else:
            cut = gen_cut(x, self.variable_type, n=self.cut_cnt, mthd=self.cut_mthd, precision=4)
            cross, cut = gen_cross(x, y, cut, self.variable_type)
            if (cross.loc[cross.index != -1, 0] == 0).all() or (cross.loc[cross.index != -1, 1] == 0).all():
                return self
            cross, cut = self._lowpct_zero_merge(cross, cut)
            bin_dic = self._gen_comb_bins(cross, cut, verbose)
            self.bin_dic = bin_dic
            best_dic = self._select_best(bin_dic)
            self.best_dic = best_dic
        return self

    def transform(self, x):
        """单变量应用."""
        trns = pd.DataFrame()
        for i in self.best_dic:
            shape = self.best_dic[i]['shape']
            bin_cnt = self.best_dic[i]['bin_cnt']
            cut = self.best_dic[i]['cut']
            woe = self.best_dic[i]['WOE']
            name = '_'.join([str(x.name), str(shape), str(bin_cnt)])
            x_trns = apply_woe(x, cut, woe, self.variable_type)
            x_trns.name = name
            trns = pd.concat([trns, x_trns], axis=1)
        return trns

    def transform_bin(self, x):
        """单变量应用返回bin."""
        trns = pd.DataFrame()
        for i in self.best_dic:
            shape = self.best_dic[i]['shape']
            bin_cnt = self.best_dic[i]['bin_cnt']
            cut = self.best_dic[i]['cut']
            name = '_'.join([str(x.name), str(shape), str(bin_cnt)])
            x_trns = apply_cut_bin(x, cut, self.variable_type)
            x_trns.name = name
            trns = pd.concat([trns, x_trns], axis=1)
        return trns


class ExploreVarBinning:
    """探索性分箱."""

    def __init__(self, cut_cnt=50, thrd_PCT=0.025, thrd_n=None, verbose=True,
                 max_bin_cnt=6, I_min=3, U_min=4, cut_mthd='eqqt',
                 variable_shape=None, tolerance=0, **kwargs):
        self.cut_cnt = cut_cnt
        self.thrd_PCT = thrd_PCT
        self.thrd_n = thrd_n
        self.max_bin_cnt = max_bin_cnt
        self.I_min = I_min
        self.U_min = U_min
        self.cut_mthd = cut_mthd
        self.variable_shape = list(variable_shape) or list('IDU')
        self.variable_type = type(kwargs.get('variable_type'))
        self.tolerance = tolerance
        self.describe = kwargs.get('describe', '未知')
        self.verbose = verbose
        self.best_dic = {}
        self.bin_dic = {}
        param_in_validate(self.cut_mthd, ['eqqt', 'eqdist'], '使用了cut_mthd={0}, 仅支持cut_mthd={1}')
        param_contain_validate(self.variable_shape, ['I', 'D', 'U', 'A'], '使用了variable_shape={0}, 仅支持variable_shape={1}')

    def _lowpct_zero_merge(self, crs, cut):
        cross = crs.copy(deep=True)
        cross, pct_cut = merge_lowpct_zero(cross, cut, self.variable_type,
                                           thrd_PCT=self.thrd_PCT,
                                           thrd_n=self.thrd_n,
                                           mthd='PCT')
        cross, t_cut = merge_lowpct_zero(cross, pct_cut, self.variable_type,
                                         thrd_PCT=self.thrd_PCT,
                                         thrd_n=self.thrd_n,
                                         mthd='zero')
        return cross, t_cut

    def _gen_comb_bins(self, crs, cut, verbose=False):
        cross = crs.copy(deep=True)
        min_I = self.I_min - 1
        min_U = self.U_min - 1
        if 'U' not in self.variable_shape:
            min_cut_cnt = min_I
        elif 'I' not in self.variable_shape and 'D' not in self.variable_shape:
            min_cut_cnt = min_U
        else:
            min_cut_cnt = min(min_I, min_U)
        max_cut_cnt = self.max_bin_cnt - 1
        cut_point_list = [x for x in cross.index[:] if x != -1][1:]
        cut_point_cnt = len(cut_point_list)
        # 限定分组数的上下限
        max_cut_loops_cnt = cut_point_cnt - min_cut_cnt + 1
        min_cut_loops_cnt = max(cut_point_cnt - max_cut_cnt, 0)
        var_bin_dic = {}
        s = 0
        loops_ = range(min_cut_loops_cnt, max_cut_loops_cnt)
        bcs = [bi for loop in loops_ for bi in its.combinations(cut_point_list, loop)]
        if len(bcs) > 0:
            tqdm_options = {'bar_format': PBAR_FORMAT,
                            'total': len(bcs),
                            'desc': self.indep,
                            'disable': True}
            if verbose:
                tqdm_options.update({'disable': False})
            with make_tqdm_iterator(**tqdm_options) as progress_bar:
                for bin_idxs in bcs:
                    # 根据选取的切点合并列联表
                    merged = merge_bin_by_idx(cross, bin_idxs)
                    shape = bad_rate_shape(merged, self.I_min, self.U_min)
                    # badrate的形状符合先验形状的分箱方式保留下来
                    if pd.isna(shape) or shape not in self.variable_shape:
                        progress_bar.update()
                        continue
                    detail, iv = calwoe(merged)
                    woe = {key: val['WOE'] for key, val in detail.items() if key != -1}
                    woe_ = pd.Series(woe)
                    tol = np.min(np.abs(woe_ - woe_.shift()))
                    if tol <= self.tolerance:
                        progress_bar.update()
                        continue
                    var_bin_dic[s] = {}
                    chi, p, dof, expFreq =\
                        sps.chi2_contingency(
                                merged.loc[~merged.index.isin([-1]), :].values,
                                correction=False)
                    var_entropy = sps.entropy(pd.Series(
                        [val['all_num'] for key, val in detail.items()
                         if key != -1]))
                    var_bin_dic[s] = {
                        'detail': detail,
                        'IV': iv,
                        'flogp': -np.log(max(p, 1e-5)),
                        'entropy': var_entropy,
                        'shape': shape,
                        'bin_cnt': len(merged)-1,
                        'cut': cut_adj(cut, bin_idxs, self.variable_type),
                        'WOE': woe,
                        'describe': self.describe,
                        'quantile': self.quantile}
                    s += 1
                    progress_bar.update()
        return var_bin_dic

    def fit(self, x, y):
        """单变量训练."""
        verbose = self.verbose
        if x.dropna().nunique() <= 1:
            # if verbose:
            #     print('nunique <= 1: {}, {}'.format(x.name, list(x.unique())))
            return self
        self.dep = y.name
        self.indep = x.name
        if issubclass(self.variable_type, (vtype.Summ, vtype.Count)):
            self.quantile = np.percentile(x.dropna(), [0, 25, 50, 75, 100]).tolist()
        else:
            self.quantile = []
        cut = gen_cut(x, self.variable_type, n=self.cut_cnt, mthd=self.cut_mthd, precision=4)
        cross, cut = gen_cross(x, y, cut, self.variable_type)
        if (cross.loc[cross.index != -1, 0] == 0).all() or (cross.loc[cross.index != -1, 1] == 0).all():
            return self
        cross, cut = self._lowpct_zero_merge(cross, cut)
        bin_dic = self._gen_comb_bins(cross, cut, verbose)
        self.bin_dic = bin_dic
        return self

    def load_bins(self, bins, indep):
        """加载数据."""
        self.bin_dic = bins
        self.indep = indep
        return self

    def grid_search_best(self, verbose=True, bins_cnt=None,
                         variable_shape=None,
                         slc_mthd=['ivae', 'flogp', 'entropy', 'IV', 'minptp'],
                         tolerance=np.linspace(0.0, 0.1, 11).tolist(),
                         **kwargs):
        """网格选择最优分箱."""
        def _normalize(x):
            x_min = x.min()
            x_max = x.max()
            return (x - x_min)/(x_max - x_min)

        def _ptp(x, qt):
            if isinstance(x, dict):
                return -1 * np.ptp(x.values.value_counts())
            clip_min, clip_max = np.clip(qt, 2.5 * qt[1] - 1.5 * qt[3], 2.5 * qt[3] - 1.5 * qt[1])[[0, -1]]
            x = deepcopy(x)
            x[0] = qt[0] if clip_min > x[1] else clip_min
            x[-1] = qt[-1] if clip_max < x[-2] else clip_max
            return -1 * np.ptp(np.diff(x))

        def _caltol(woes):
            woe = {key: val for key, val in woes.items() if key != -1}
            woe_ = pd.Series(woe)
            return np.min(np.abs(woe_ - woe_.shift()))

        def _deft_null(x, dx):
            if x is None:
                return dx
            if isinstance(x, list):
                return x
            return [x]
        bins_cnt = _deft_null(bins_cnt, list(range(min(self.I_min, self.U_min), self.max_bin_cnt)))
        slc_mthd = _deft_null(slc_mthd, ['ivae', 'flogp', 'entropy', 'IV', 'minptp'])
        tolerance = [tolx for tolx in tolerance if tolx >= self.tolerance]
        variable_shape = variable_shape or self.variable_shape
        param_contain_validate(
            slc_mthd, ['flogp', 'IV', 'entropy', 'ivae', 'minptp'],
            '使用了slc_mthd={0}, 仅支持slc_mthd={1}')
        dic = self.bin_dic
        if len(dic) == 0:
            return self
        dd = pd.DataFrame.from_dict(dic, orient='index')
        dd.loc[:, 'minptp'] = dd.apply(lambda x: _ptp(x['cut'], x['quantile']), axis=1)
        dd.loc[:, 'tolerance'] = dd.apply(lambda x: _caltol(x['WOE']), axis=1)
        norm_iv = _normalize(dd.loc[:, 'IV'])
        norm_e = _normalize(dd.loc[:, 'entropy'])
        dd.loc[:, 'ivae'] = np.hypot(norm_iv, norm_e)
        prod_ = list(product(bins_cnt, variable_shape, slc_mthd, tolerance))
        if len(prod_) <= 0:
            return self
        tqdm_options = {'bar_format': PBAR_FORMAT,
                        'total': len(prod_),
                        'desc': self.indep,
                        'disable': True}
        if self.verbose:
            tqdm_options.update({'disable': False})
        best_dd = {}
        exists = {}
        with make_tqdm_iterator(**tqdm_options) as progress_bar:
            for prod in prod_:
                cnt, shape, mthd, tol = prod
                tem_dd = dd.loc[(dd.loc[:, 'bin_cnt'] == cnt)
                                & (dd.loc[:, 'shape'] == shape)
                                & (dd.loc[:, 'tolerance'] >= tol), :]
                if len(tem_dd) <= 0:
                    progress_bar.update()
                    continue
                parm = (('bins_cnt', cnt), ('variable_shape', shape),
                        ('slc_mthd', mthd), ('tolerance', tol))
                tem_dic = tem_dd.sort_values(by=mthd, ascending=False)\
                    .iloc[[0], :].to_dict(orient='index')
                tem_dic_key = list(tem_dic.keys())[0]
                val = list(tem_dic.values())[0]
                if tem_dic_key in exists.keys():
                    del best_dd[exists[tem_dic_key]]
                exists.update({tem_dic_key: parm})
                best_dd.update({parm: val})
                progress_bar.update()
        self.best_dic = best_dd
        return self

    def plot_best(self):
        """图示."""
        if len(self.best_dic) <= 0:
            return self
        details = pd.DataFrame()
        for best_key, best_val in self.best_dic.items():
            detail = pd.DataFrame.from_dict(best_val['detail'], orient='index')
            cut_str = cut_to_interval(best_val['cut'], self.variable_type)
            cut_str.update({-1: 'NaN'})
            name = '{}\n{}'.format(self.indep, str(best_key))
            detail.loc[:, 'Bound'] = pd.Series(cut_str)
            detail.loc[:, 'var'] = name
            detail.loc[:, 'describe'] = name
            details = pd.concat([details, detail])
        plot_bin(details)
        return self


# =============================================================================
# class VarBinning:
#     """单个变量分箱.
#
#     Input_
#         subjective_cut: [list]|{dict}
#
#     Return_
#         bin_dic: {_id: {detail: {},
#                         IV: iv,
#                         WOE: {},
#                         bin_cnt: cnt,
#                         cut: list,
#                         describle: str,
#                         entropy: entropy,
#                         flogp: -log(p),
#                         num_inflection: -n_inflection,
#                         shape: U/I/D}}}}
#         best_dic: upon bin_dic
#     """
#
#     def __init__(self, subjective_cut=None, **kwargs):
#         self.variable_type = type(kwargs.get('variable_type'))
#         self.describe = kwargs.get('describe', '未知')
#         self.subjective_cut = subjective_cut
#         self.best_dic = {}
#
#     def fit(self, x, y):
#         """单变量训练."""
#         if x.dropna().nunique() <= 1:
#             # if verbose:
#             #     print('nunique <= 1: {}, {}'.format(x.name, list(x.unique())))
#             return self
#         self.dep = y.name
#         self.indep = x.name
#         if issubclass(self.variable_type, (vtype.Summ, vtype.Count)):
#             self.quantile = np.percentile(x.dropna(), [0, 25, 50, 75, 100]).tolist()
#         else:
#             self.quantile = []
#         cut = self.subjective_cut
#         cross, t_cut = gen_cross(x, y, cut, self.variable_type)
#         detail, iv = calwoe(cross)
#         woe = {key: val['WOE'] for key, val in detail.items() if key != -1}
#         chi, p, dof, expFreq =\
#             sps.chi2_contingency(
#                     cross.loc[~cross.index.isin([-1]), :].values,
#                     correction=False)
#         var_entropy = sps.entropy(pd.Series(
#             [val['all_num'] for key, val in detail.items()
#              if key != -1]))
#         shape = bad_rate_shape(cross, 0, 0)
#         self.bin_dic = {0: {
#             'detail': detail,
#             'IV': iv,
#             'flogp': -np.log(max(p, 1e-5)),
#             'entropy': var_entropy,
#             'shape': shape,
#             'bin_cnt': len(cross)-1,
#             'cut': cut,
#             'WOE': woe,
#             'describe': self.describe,
#             'quantile': self.quantile}
#             }
#         self.best_dic = deepcopy(self.bin_dic)
#         return self
#
#     def transform(self, x):
#         """单变量应用."""
#         trns = pd.DataFrame()
#         for i in self.best_dic:
#             shape = self.best_dic[i]['shape']
#             bin_cnt = self.best_dic[i]['bin_cnt']
#             cut = self.best_dic[i]['cut']
#             woe = self.best_dic[i]['WOE']
#             name = '_'.join([str(x.name), str(shape), str(bin_cnt)])
#             x_trns = apply_woe(x, cut, woe, self.variable_type)
#             x_trns.name = name
#             trns = pd.concat([trns, x_trns], axis=1)
#         return trns
#
#     def transform_bin(self, x):
#         """单变量应用返回bin."""
#         trns = pd.DataFrame()
#         for i in self.best_dic:
#             shape = self.best_dic[i]['shape']
#             bin_cnt = self.best_dic[i]['bin_cnt']
#             cut = self.best_dic[i]['cut']
#             name = '_'.join([str(x.name), str(shape), str(bin_cnt)])
#             x_trns = apply_cut_bin(x, cut, self.variable_type)
#             x_trns.name = name
#             trns = pd.concat([trns, x_trns], axis=1)
#         return trns
# =============================================================================

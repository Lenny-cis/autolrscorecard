# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:51:04 2021

@author: linjianing
"""


import os
import numpy as np
import pandas as pd
import pickle as pkl
from copy import deepcopy
from openpyxl import load_workbook
from autolrscorecard.woebin.varbinning import VarBinning, ExploreVarBinning
from autolrscorecard.utils.woebin_utils import (
    make_tqdm_iterator, cut_to_interval)

PBAR_FORMAT = "Possible: {total} | Elapsed: {elapsed} | Progress: {l_bar}{bar}"


class ExploreWOEBinning:
    """特征全集探索性分箱."""

    def __init__(self, entity):
        self.entity = entity
        self.data = {'bins_set': {}}

    def fit(self, verbose=True, **kwargs):
        """训练."""
        print('*'*40, 'EXPLORE BINNING', '*'*40)
        entity = self.entity
        variable_options = entity.variable_options
        y = entity.pipe_y
        X = entity.pipe_X
        bins_set = {}
        for x_name in X.columns:
            vop = deepcopy(kwargs)
            vop.update(variable_options.get(x_name, {}))
            x_binning = ExploreVarBinning(**vop)
            x_binning.fit(X.loc[:, x_name], y)
            if x_binning.bins_set != {}:
                bins_set.update({x_name: x_binning})
        self.bins_set = bins_set
        return self

    def grid_search_best(self, search_params={}, verbose=True,
                         search_vars=None, **kwargs):
        """网格选择特征全集最优分箱."""
        print('*'*40, 'GRID SEARCH', '*'*40)
        if search_vars is None:
            search_vars = self.bins_set.keys()
        elif not isinstance(search_vars, list):
            search_vars = [search_vars]
        for key, val in self.bins_set.items():
            if key not in search_vars:
                continue
            spms = deepcopy(kwargs)
            spms.update(search_params.get(key, {}))
            val.grid_search_best(verbose=verbose, **spms)
        return self

    def manual_select(self, best_keys):
        """人工选择."""
        for x_name, best_key in best_keys.items():
            self.bins_set[x_name].manual_select(best_key)
        return self

    def fast_search_best(self, search_params={}, verbose=True,
                         search_vars=None, **kwargs):
        """快速选择特征全集最优分箱."""
        print('*'*40, 'FAST SEARCH', '*'*40)
        if search_vars is None:
            search_vars = self.bins_set.keys()
        elif not isinstance(search_vars, list):
            search_vars = [search_vars]
        tqdm_options = {'bar_format': PBAR_FORMAT,
                        'total': len(self.bins_set),
                        'disable': True}
        if verbose:
            tqdm_options.update({'disable': False})
        with make_tqdm_iterator(**tqdm_options) as progress_bar:
            for key, val in self.bins_set.items():
                if key not in search_vars:
                    progress_bar.update()
                    continue
                spms = deepcopy(kwargs)
                spms.update(search_params.get(key, {}))
                val.fast_search_best(verbose=verbose, **spms)
                progress_bar.update()
        self.entity.steps.update({'binning': self.selected_best})
        return self

    def transform(self, entity=None, retbin=False, verbose=True):
        """执行应用."""
        if entity is None:
            entity = self.entity
        X = entity.pipe_X
        tqdm_options = {'bar_format': PBAR_FORMAT,
                        'total': len(self.best_bins),
                        'disable': True}
        if verbose:
            tqdm_options.update({'disable': False})
        X_trns = entity.df.loc[:, [entity.target]]
        with make_tqdm_iterator(**tqdm_options) as progress_bar:
            for x_name, x_binning in self.bins_set.items():
                x_trns = x_binning.transform(
                    X.loc[:, x_name], retbin)
                X_trns = pd.concat([X_trns, x_trns], axis=1)
                progress_bar.update()
        entity.pipe_X = X_trns
        entity.update_data()
        return self

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
        best_bins = {}
        for x, bins in self.bins_set.items():
            best_bins.update({x: bins.best_bins})
        return best_bins

    @property
    def selected_best(self):
        """已选择的最优分箱."""
        selected_best_ = {}
        for x_name, x_binning in self.bins_set.items():
            x_selected_best = x_binning.selected_best
            if len(x_selected_best) == 0:
                continue
            selected_best_.update({x_name: x_selected_best})
        return selected_best_

    def plot_best(self, plot_vars=None):
        """图示."""
        print('*'*40, 'PLOT BEST', '*'*40)
        if plot_vars is None:
            plot_vars = self.bins_set.keys()
        elif not isinstance(plot_vars, list):
            plot_vars = [plot_vars]
        for key, val in self.bins_set.items():
            if key not in plot_vars:
                continue
            x_binning = val
            x_binning.plot_best()
        return self

    def dump(self, save_file):
        """保存类."""
        print('*'*40, 'SAVE DATA', '*'*40)
        dm = {}
        for k, v in self.data.items():
            if k == 'bins_set':
                tv = {}
                for vk, vv in v.items():
                    tv.update({vk: vv.to_dict()})
                dm.update({k: tv})
            else:
                dm.update({k: v})
        with open(save_file, 'wb') as f:
            pkl.dump(dm, f)

    def load(self, save_file):
        """加载类."""
        with open(save_file, 'rb') as f:
            dm = pkl.load(f)
        for k, v in dm.items():
            if k == 'bins_set':
                tv = {}
                for vk, vv in v.items():
                    tv.update({vk: ExploreVarBinning.load(vv)})
                setattr(self, k, tv)
            else:
                setattr(self, k, v)
        return self

    def output(self):
        """输出报告."""
        pass

    def report(self, file):
        df = pd.DataFrame()
        variable_options = self.entity.variable_options
        for x, dic in self.selected_best.items():
            if dic == {}:
                continue
            detail = pd.DataFrame.from_dict(dic['detail'], orient='index')
            cut = deepcopy(dic['cut'])
            cut_str = cut_to_interval(
                cut,
                type(variable_options.get(x).get('variable_type')))
            cut_str.update({-1: 'NaN'})
            detail.loc[:, 'Bound'] = pd.Series(cut_str)
            detail.loc[:, 'SUMIV'] = dic['IV']
            detail.loc[:, 'entropy'] = dic['entropy']
            detail.loc[:, 'flogp'] = dic['flogp']
            detail.loc[:, 'shape'] = dic['shape']
            detail.loc[:, 'var'] = '_'.join(
                [x, str(dic['shape']), str(dic['bin_cnt'])])
            detail.loc[:, 'describe'] = dic.get('describe', '未知')
            detail = detail.loc[:, [
                'var', 'describe', 'Bound',
                'all_num', 'event_num', 'event_rate', 'WOE', 'shape',
                'IV', 'SUMIV', 'entropy', 'flogp']]
            df = pd.concat([df, detail])
        if not os.path.exists(file):
            with pd.ExcelWriter(file) as writer:
                df.to_excel(writer, sheet_name='binning')
        else:
            with pd.ExcelWriter(file, engine='openpyxl') as writer:
                book = load_workbook(file)
                writer.book = book
                df.to_excel(writer, sheet_name='binning')
        return self


class WOEBinning:
    """特征全集分箱."""

    def __init__(self, entity):
        self.entity = entity
        self.data = {'bins_set': {}, 'best_bins': {}}

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
        best_bins = {}
        for x, bins in self.bins_set.items():
            best_bins.update({x: bins.best_bins})
        return best_bins

    def fit(self, verbose=True, **kwargs):
        """训练."""
        entity = self.entity
        variable_options = entity.variable_options
        y = entity.df.loc[:, entity.target]
        X = entity.df.loc[:, entity.independents]
        bins_set = {}
        for x_name in X.columns:
            vop = deepcopy(kwargs)
            vop.update(variable_options.get(x_name, {}))
            x_binning = VarBinning(**vop)
            x_binning.fit(X.loc[:, x_name], y)
            if x_binning.bin_dic != {}:
                bins_set.update({x_name: x_binning})
        self.bins_set = bins_set
        return self

    def transform(self, verbose=True):
        """应用."""
        entity = self.entity
        X = entity.df.loc[:, entity.independents]
        X_trns = X.copy(deep=True)
        X_names = X.columns
        tqdm_options = {'bar_format': PBAR_FORMAT,
                        'total': len(self.best_bins),
                        'disable': True}
        if verbose:
            tqdm_options.update({'disable': False})
        with make_tqdm_iterator(**tqdm_options) as progress_bar:
            for x_name, x_binning in self.bins_set.items():
                x_trns = x_binning.transform(X.loc[:, x_name])
                X_trns = pd.concat([X_trns, x_trns], axis=1)
                progress_bar.update()
        X_trns.drop(X_names, axis=1, inplace=True)
        return X_trns

    def transform_bin(self, X):
        """应用."""
        entity = self.entity
        X = entity.df.loc[:, entity.independents]
        X_trns = X.copy(deep=True)
        X_names = X.columns
        tqdm_options = {'bar_format': PBAR_FORMAT,
                        'total': len(self.best_bins),
                        'disable': True}
        if self.verbose:
            tqdm_options.update({'disable': False})
        with make_tqdm_iterator(**tqdm_options) as progress_bar:
            for x_name, x_binning in self.bins_set.items():
                x_trns = x_binning.transform_bin(X.loc[:, x_name])
                X_trns = pd.concat([X_trns, x_trns], axis=1)
                progress_bar.update()
        X_trns.drop(X_names, axis=1, inplace=True)
        return X_trns

    def output(self):
        """输出报告."""
        out = pd.DataFrame()
        for x, bdict in self.best_bins.items():
            if bdict == {}:
                continue
            for k, dic in bdict.items():
                detail = pd.DataFrame.from_dict(dic['detail'], orient='index')
                cut = deepcopy(dic['cut'])
                cut_str = cut_to_interval(
                    cut,
                    type(self.variable_options.get(x).get('variable_type')))
                cut_str.update({-1: 'NaN'})
                detail.loc[:, 'Bound'] = pd.Series(cut_str)
                detail.loc[:, 'SUMIV'] = dic['IV']
                detail.loc[:, 'entropy'] = dic['entropy']
                detail.loc[:, 'flogp'] = dic['flogp']
                detail.loc[:, 'shape'] = dic['shape']
                detail.loc[:, '_id'] = k
                detail.loc[:, 'var'] = '_'.join(
                    [x, str(dic['shape']), str(dic['bin_cnt'])])
                detail.loc[:, 'describe'] = dic.get('describe', '未知')
                detail = detail.loc[:, [
                    '_id', 'var', 'describe', 'Bound',
                    'all_num', 'event_num', 'event_rate', 'WOE', 'shape',
                    'IV', 'SUMIV', 'entropy', 'flogp']]
                out = pd.concat([out, detail])
        return out


# class WOEBinning:
#     """特征全集分箱."""

#     def __init__(self, variable_options={}, verbose=True, **kwargs):
#         self.variable_options = variable_options
#         self.kwargs = kwargs
#         self.verbose = verbose
#         self.best_bins = {}
#         self.bin_dic = {}

#     def fit(self, X, y):
#         """训练."""
#         for x_name in X.columns:
#             vop = deepcopy(self.kwargs)
#             vop.update(self.variable_options.get(x_name, {}))
#             x_binning = VarBinning(**vop)
#             x_binning.fit(X.loc[:, x_name], y)
#             if x_binning.bin_dic != {}:
#                 self.bin_dic.update({x_name: x_binning})
#                 self.best_bins.update({x_name: x_binning.best_dic})
#         return self

#     def transform(self, X):
#         """应用."""
#         X_trns = X.copy(deep=True)
#         X_names = X.columns
#         tqdm_options = {'bar_format': PBAR_FORMAT,
#                         'total': len(self.best_bins),
#                         'disable': True}
#         if self.verbose:
#             tqdm_options.update({'disable': False})
#         with make_tqdm_iterator(**tqdm_options) as progress_bar:
#             for x_name, x_binning in self.bin_dic.items():
#                 x_trns = x_binning.transform(X.loc[:, x_name])
#                 X_trns = pd.concat([X_trns, x_trns], axis=1)
#                 progress_bar.update()
#         X_trns.drop(X_names, axis=1, inplace=True)
#         return X_trns

#     def transform_bin(self, X):
#         """应用."""
#         X_trns = X.copy(deep=True)
#         X_names = X.columns
#         tqdm_options = {'bar_format': PBAR_FORMAT,
#                         'total': len(self.bin_dic),
#                         'disable': True}
#         if self.verbose:
#             tqdm_options.update({'disable': False})
#         with make_tqdm_iterator(**tqdm_options) as progress_bar:
#             for x_name, x_binning in self.bin_dic.items():
#                 x_trns = x_binning.transform_bin(X.loc[:, x_name])
#                 X_trns = pd.concat([X_trns, x_trns], axis=1)
#                 progress_bar.update()
#         X_trns.drop(X_names, axis=1, inplace=True)
#         return X_trns

#     def output(self):
#         """输出报告."""
#         out = pd.DataFrame()
#         for x, bdict in self.best_bins.items():
#             if bdict == {}:
#                 continue
#             for k, dic in bdict.items():
#                 detail = pd.DataFrame.from_dict(dic['detail'], orient='index')
#                 cut = deepcopy(dic['cut'])
#                 cut_str = cut_to_interval(cut, type(self.variable_options.get(x).get('variable_type')))
#                 cut_str.update({-1: 'NaN'})
#                 detail.loc[:, 'Bound'] = pd.Series(cut_str)
#                 detail.loc[:, 'SUMIV'] = dic['IV']
#                 detail.loc[:, 'entropy'] = dic['entropy']
#                 detail.loc[:, 'flogp'] = dic['flogp']
#                 detail.loc[:, 'shape'] = dic['shape']
#                 detail.loc[:, '_id'] = k
#                 detail.loc[:, 'var'] = '_'.join([x, str(dic['shape']), str(dic['bin_cnt'])])
#                 detail.loc[:, 'describe'] = dic.get('describe', '未知')
#                 detail = detail.loc[:, ['_id', 'var', 'describe', 'Bound',
#                                         'all_num', 'event_num', 'event_rate', 'WOE', 'shape',
#                                         'IV', 'SUMIV', 'entropy', 'flogp']]
#                 out = pd.concat([out, detail])
#         return out


# class ExploreWOEBinning:
#     """特征全集探索性分箱."""

#     def __init__(self, variable_options={}, verbose=True, **kwargs):
#         self.variable_options = variable_options
#         self.kwargs = kwargs
#         self.verbose = verbose
#         self.bin_dic = {}
#         self.best_bins = {}

#     def fit(self, X, y):
#         """训练."""
#         print('*'*40, 'EXPLORE BINNING', '*'*40)
#         for x_name in X.columns:
#             vop = deepcopy(self.kwargs)
#             vop.update(self.variable_options.get(x_name, {}))
#             x_binning = ExploreVarBinning(**vop)
#             x_binning.fit(X.loc[:, x_name], y)
#             if x_binning.bin_dic != {}:
#                 self.bin_dic.update({x_name: x_binning})
#         return self

#     # def load_bins(self, bins):
#     #     """加载数据."""
#     #     for
#     #     self.bin_dic = bins
#     #     return self

#     def grid_search_best(self, search_params={}, verbose=True,
#                          search_vars=None, **kwargs):
#         """网格选择特征全集最优分箱."""
#         print('*'*40, 'GRID SEARCH', '*'*40)
#         if search_vars is None:
#             search_vars = self.bin_dic.keys()
#         elif not isinstance(search_vars, list):
#             search_vars = [search_vars]
#         for key, val in self.bin_dic.items():
#             if key not in search_vars:
#                 continue
#             spms = deepcopy(kwargs)
#             spms.update(search_params.get(key, {}))
#             x_binning = val
#             x_binning.grid_search_best(verbose=verbose, **spms)
#         return self

#     def plot_best(self, plot_vars=None):
#         """图示."""
#         print('*'*40, 'PLOT BEST', '*'*40)
#         if plot_vars is None:
#             plot_vars = self.bin_dic.keys()
#         elif not isinstance(plot_vars, list):
#             plot_vars = [plot_vars]
#         for key, val in self.bin_dic.items():
#             if key not in plot_vars:
#                 continue
#             x_binning = val
#             x_binning.plot_best()
#         return self

#     def dump(self, save_file):
#         """保存类."""
#         print('*'*40, 'SAVE DATA', '*'*40)
#         dm = {}
#         for k, v in vars(self).items():
#             if k == 'bin_dic':
#                 tv = {}
#                 for vk, vv in v.items():
#                     tv.update({vk: vv.to_dict()})
#                 dm.update({k: tv})
#             else:
#                 dm.update({k: v})
#         with open(save_file, 'wb') as f:
#             pkl.dump(dm, f)

#     @classmethod
#     def load(cls, save_file):
#         """加载类."""
#         obj = cls.__new__(cls)
#         with open(save_file, 'rb') as f:
#             dm = pkl.load(f)
#         for k, v in dm.items():
#             if k == 'bin_dic':
#                 tv = {}
#                 for vk, vv in v.items():
#                     tv.update({vk: ExploreVarBinning.load(vv)})
#                 setattr(obj, k, tv)
#             else:
#                 setattr(obj, k, v)
#         return obj

#     def output(self):
#         pass
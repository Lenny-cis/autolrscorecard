# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:51:04 2021

@author: linjianing
"""


import numpy as np
import pandas as pd
import pickle as pkl
from copy import deepcopy
from .varbinning import VarBinning, ExploreVarBinning
from utils.woebin_utils import make_tqdm_iterator, cut_to_interval

PBAR_FORMAT = "Possible: {total} | Elapsed: {elapsed} | Progress: {l_bar}{bar}"


class WOEBinning:
    """特征全集分箱."""

    def __init__(self, variable_options={}, verbose=True, **kwargs):
        self.variable_options = variable_options
        self.kwargs = kwargs
        self.verbose = verbose
        self.bin_dic = {}
        self.best_bins = {}

    def fit(self, X, y):
        """训练."""
        for x_name in X.columns:
            vop = deepcopy(self.kwargs)
            vop.update(self.variable_options.get(x_name, {}))
            x_binning = VarBinning(**vop)
            x_binning.fit(X.loc[:, x_name], y)
            if x_binning.bin_dic != {}:
                self.bin_dic.update({x_name: x_binning})
                self.best_bins.update({x_name: x_binning.best_dic})
        return self

    def transform(self, X):
        """应用."""
        X_trns = X.copy(deep=True)
        X_names = X.columns
        tqdm_options = {'bar_format': PBAR_FORMAT,
                        'total': len(self.best_bins),
                        'disable': True}
        if self.verbose:
            tqdm_options.update({'disable': False})
        with make_tqdm_iterator(**tqdm_options) as progress_bar:
            for x_name, x_binning in self.bin_dic.items():
                x_trns = x_binning.transform(X.loc[:, x_name])
                X_trns = pd.concat([X_trns, x_trns], axis=1)
                progress_bar.update()
        X_trns.drop(X_names, axis=1, inplace=True)
        return X_trns

    def transform_bin(self, X):
        """应用."""
        X_trns = X.copy(deep=True)
        X_names = X.columns
        tqdm_options = {'bar_format': PBAR_FORMAT,
                        'total': len(self.best_bins),
                        'disable': True}
        if self.verbose:
            tqdm_options.update({'disable': False})
        with make_tqdm_iterator(**tqdm_options) as progress_bar:
            for x_name, x_binning in self.bin_dic.items():
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
                cut_str = cut_to_interval(cut, type(self.variable_options.get(x).get('variable_type')))
                cut_str.update({-1: 'NaN'})
                detail.loc[:, 'Bound'] = pd.Series(cut_str)
                detail.loc[:, 'SUMIV'] = dic['IV']
                detail.loc[:, 'entropy'] = dic['entropy']
                detail.loc[:, 'flogp'] = dic['flogp']
                detail.loc[:, 'shape'] = dic['shape']
                detail.loc[:, '_id'] = k
                detail.loc[:, 'var'] = '_'.join([x, str(dic['shape']), str(dic['bin_cnt'])])
                detail.loc[:, 'describe'] = dic.get('describe', '未知')
                detail = detail.loc[:, ['_id', 'var', 'describe', 'Bound',
                                        'all_num', 'event_num', 'event_rate', 'WOE', 'shape',
                                        'IV', 'SUMIV', 'entropy', 'flogp']]
                out = pd.concat([out, detail])
        return out


class ExploreWOEBinning:
    """特征全集探索性分箱."""

    def __init__(self, variable_options={}, verbose=True, **kwargs):
        self.variable_options = variable_options
        self.kwargs = kwargs
        self.verbose = verbose
        self.bin_dic = {}
        self.best_bins = {}

    def fit(self, X, y):
        """训练."""
        print('*'*40, 'EXPLORE BINNING', '*'*40)
        for x_name in X.columns:
            vop = deepcopy(self.kwargs)
            vop.update(self.variable_options.get(x_name, {}))
            x_binning = ExploreVarBinning(**vop)
            x_binning.fit(X.loc[:, x_name], y)
            if x_binning.bin_dic != {}:
                self.bin_dic.update({x_name: x_binning})
        return self

    # def load_bins(self, bins):
    #     """加载数据."""
    #     for
    #     self.bin_dic = bins
    #     return self

    def grid_search_best(self, search_params={}, verbose=True, **kwargs):
        """网格选择特征全集最优分箱."""
        print('*'*40, 'GRID SEARCH', '*'*40)
        for key, val in self.bin_dic.items():
            spms = deepcopy(kwargs)
            spms.update(search_params.get(key, {}))
            x_binning = val
            x_binning.grid_search_best(verbose=verbose, **spms)
        return self

    def plot_best(self):
        """图示."""
        print('*'*40, 'PLOT BEST', '*'*40)
        for key, val in self.bin_dic.items():
            x_binning = val
            x_binning.plot_best()
        return self

    def dump(self, save_file):
        """保存类."""
        print('*'*40, 'SAVE DATA', '*'*40)
        with open(save_file, 'wb') as f:
            pkl.dump(self, f)

    @staticmethod
    def load(save_file):
        """加载类."""
        with open(save_file, 'rb') as f:
            return pkl.load(f)

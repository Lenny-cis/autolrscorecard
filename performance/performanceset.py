# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:10:04 2021

@author: linjianing
"""


from .singleseries import missing_ratio, concentration_ratio, num_unique
from .withflag import IV, KS

defualt_threshold = {'nomissing': 0.05,
                     'nonconcentration': 0.05,
                     'nunique': 0,
                     'iv': 0.01,
                     'ks': 0.01,
                     'auc': 0.5}


class DescribeRatio:
    """缺失率."""

    def __init__(self, var_options={}, slc_threshold=defualt_threshold):
        self.slc_threshold = slc_threshold
        self.describe_ratio = {}
        self.describe_vars = []

    def fit(self, X, y):
        """训练."""
        for x_name in X.columns:
            missing_ratio_ = missing_ratio(X.loc[:, x_name])
            concentration_ratio_ = concentration_ratio(X.loc[:, x_name])
            num_unique_ = num_unique(X.loc[:, x_name])
            if (self.slc_threshold.get('nomissing') <= 1-missing_ratio_
                and self.slc_threshold.get('nonconcentration') <= 1-concentration_ratio_
                    and self.slc_threshold.get('nunique') <= num_unique_):
                self.describe_vars.append(x_name)
            self.describe_ratio.update({x_name: {'nonmissing': 1-missing_ratio_,
                                                 'nonconcentration': 1-concentration_ratio_,
                                                 'nunique': num_unique_}})
        return self

    def transform(self, X):
        """应用."""
        return X.loc[:, self.describe_vars].copy(deep=True)

    def output(self):
        """输出报告."""
        pass


class IVSet:
    """变量预测力."""

    def __init__(self, variable_options={}, slc_threshold=defualt_threshold):
        self.slc_threshold = slc_threshold
        self.iv_vars = []
        self.ivset = {}
        self.variable_options = variable_options

    def fit(self, X, y):
        """训练."""
        for x_name in X.columns:
            var_type = self.variable_options.get(x_name)['variable_type']
            iv = IV(X.loc[:, x_name], y, var_type)
            self.ivset.update({x_name: iv})
            if self.slc_threshold.get('iv') <= iv:
                self.iv_vars.append(x_name)
        return self

    def transform(self, X):
        """应用."""
        return X.loc[:, self.iv_vars].copy(deep=True)


class KSAUCset:
    """变量最大区分力."""

    def __init__(self, variable_options={}, slc_threshold=defualt_threshold):
        self.slc_threshold = slc_threshold
        self.ksaucset = {}
        self.ksauc_vars = []
        self.variable_options = variable_options

    def fit(self, X, y):
        """训练."""
        for x_name in X.columns:
            var_type = self.variable_options.get(x_name)['variable_type']
            ks, auc = KS(X.loc[:, x_name], y, var_type)
            self.ksaucset.update({x_name: {'ks': ks, 'auc': auc}})
            print(self.slc_threshold)
            if self.slc_threshold.get('ks') <= ks and self.slc_threshold.get('auc') <= auc:
                self.ksauc_vars.append(x_name)
        return self

    def transform(self, X):
        """应用."""
        return X.loc[:, self.ksauc_vars].copy(deep=True)

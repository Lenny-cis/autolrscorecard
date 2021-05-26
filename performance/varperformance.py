# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:27:45 2021

@author: linjianing
"""


import os
import pandas as pd
from openpyxl import load_workbook
from autolrscorecard.performance.singleseries import(
    missing_ratio, concentration_ratio, num_unique)
from autolrscorecard.performance.withflag import IV, KS

defualt_threshold = {
    'nomissing': 0.05, 'nonconcentration': 0.05, 'nunique': 0, 'IV': 0.01,
    'KS': 0.01, 'AUC': 0.5
    }


class VarPerformance:
    def __init__(self, entity):
        self.entity = entity
        self.var_performance = {}

    def fit(self, thresholds=defualt_threshold):
        self.thresholds = thresholds
        entity = self.entity
        y = entity.pipe_y
        X = entity.pipe_X
        nm_thrd = thresholds.get('nomissing', 0)
        nc_thrd = thresholds.get('nonconcentration', 0)
        nq_thrd = thresholds.get('nunique', 0)
        iv_thrd = thresholds.get('IV', 0)
        ks_thrd = thresholds.get('KS', 0)
        auc_thrd = thresholds.get('AUC', 0)
        for x_name in X.columns:
            missing_ratio_ = missing_ratio(X.loc[:, x_name])
            concentration_ratio_ = concentration_ratio(X.loc[:, x_name])
            num_unique_ = num_unique(X.loc[:, x_name])
            var_type = type(entity.variable_options.get(x_name)
                            ['variable_type'])
            iv = IV(X.loc[:, x_name], y, var_type)
            ks, auc = KS(X.loc[:, x_name], y, var_type)
            ban = None
            if nm_thrd >= 1- missing_ratio_:
                ban = 'nomissing'
            elif nc_thrd >= 1 - concentration_ratio_:
                ban = 'nonconcentration'
            elif nq_thrd >= num_unique_:
                ban = 'nunique'
            elif iv_thrd >= iv:
                ban = 'IV'
            elif ks_thrd >= ks:
                ban = 'KS'
            elif auc_thrd >= auc:
                ban = 'AUC'
            self.var_performance.update({x_name: {
                'nonmissing': 1-missing_ratio_,
                'nonconcentration': 1-concentration_ratio_,
                'nunique': num_unique_,
                'IV': iv,
                'KS': ks,
                'AUC': auc,
                'ban': ban}})
        entity.steps.update({'varperf': self.var_performance})
        return self

    def transform(self, entity=None):
        if entity is None:
            entity = self.entity
        ban = pd.DataFrame.from_dict(self.var_performance, orient='index')
        ban_vars = ban.loc[ban.loc[:, 'ban'].notna(), :].index.tolist()
        entity.pipe_X = entity.pipe_X.drop(ban_vars, axis=1)
        entity.update_data()
        return self

    def report(self, file):
        df = pd.DataFrame.from_dict(self.var_performance, orient='index')
        if not os.path.exists(file):
            with pd.ExcelWriter(file) as writer:
                df.to_excel(writer, sheet_name='varperf')
        else:
            with pd.ExcelWriter(file, engine='openpyxl') as writer:
                book = load_workbook(file)
                writer.book = book
                df.to_excel(writer, sheet_name='varperf')
        return self

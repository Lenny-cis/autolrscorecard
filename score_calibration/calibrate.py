# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:16:18 2021

@author: linjianing
"""


import os
import pandas as pd
import numpy as np
from copy import deepcopy
from openpyxl import load_workbook
from autolrscorecard.utils.woebin_utils import (
    cut_to_interval)

class Calibrator:
    def __init__(self, entity):
        self.entity = entity

    def fit(self, prep, pdo, base_odds, base_score, clip):
        entity = self.entity
        coef = pd.DataFrame.from_dict(entity.steps['LRCust'], orient='index')
        binnings = entity.steps['binning']
        binnings = {k: v for k, v in binnings.items() if k in coef.index}
        y = entity.pipe_y
        loddsoff = np.log(prep/(1-prep)) - np.log(sum(y) / (len(y) - sum(y)))
        B = pdo / np.log(2)
        A = base_score + B * np.log(base_odds)
        coef.loc['const', 'Coef.'] = coef.loc['const', 'Coef.'] + loddsoff
        coef.loc[:, 'Coef.'] = -1 * B * coef.loc[:, 'Coef.']
        coef.loc['const', 'Coef.'] = coef.loc['const', 'Coef.'] + A
        const = coef.loc['const', 'Coef.']
        coef = coef.loc[coef.index != 'const', :]
        coef_mapping = coef.loc[:, 'Coef.'].to_dict()
        n_const = const / coef.shape[0]
        df = pd.DataFrame()
        for x, dic in binnings.items():
            if dic == {}:
                continue
            detail = pd.DataFrame.from_dict(dic['detail'], orient='index')
            cut = deepcopy(dic['cut'])
            cut_str = cut_to_interval(
                cut,
                type(entity.variable_options.get(x).get('variable_type')))
            cut_str.update({-1: 'NaN'})
            detail.loc[:, 'Bound'] = pd.Series(cut_str)
            detail.loc[:, 'var'] = x
            detail.loc[:, 'describe'] = dic.get('describe', '未知')
            detail.loc[:, 'WOE'] = detail.loc[:, 'WOE'] * coef_mapping[x]\
                + n_const
            detail = detail.loc[:, [
                'var', 'describe', 'Bound',
                'all_num', 'event_num', 'event_rate', 'WOE']]
            df = pd.concat([df, detail])
        df.rename(columns={'WOE': 'BIN_SCORE'}, inplace=True)
        w_ = df.groupby('var')['BIN_SCORE'].agg(lambda x: x.max() - x.min())
        w_ = w_ / w_.sum() * 100
        df.loc[:, 'weight'] = df.apply(lambda x: w_.loc[x['var']], axis=1)
        df.reset_index(inplace=True)
        pipe_X = entity.pipe_X
        pipe_cols = pipe_X.columns
        rename_mapping = {k: '_'.join([k, 'score'])
                          for k in pipe_cols}
        pipe_X_score = pd.DataFrame(columns=pipe_cols,
                                    index=pipe_X.index)
        for col in pipe_cols:
            pipe_X_score.loc[:, col] = pipe_X.loc[:, col] * coef_mapping[col]\
                + n_const
        pipe_X_score.loc[:, 'score'] = np.clip(pipe_X_score.sum(axis=1),
                                               clip[0], clip[-1])
        entity.steps.update({'calibrate': df.to_dict(orient='index')})
        self.bin_score = df
        entity.pipe_X = pipe_X_score
        self.score_df = pd.concat([
            entity.df.loc[:, pipe_cols],
            entity.pipe_X.rename(columns=rename_mapping)], axis=1)
        return self

    def report(self, file):
        df = self.bin_score
        score_df = self.score_df
        if not os.path.exists(file):
            with pd.ExcelWriter(file) as writer:
                df.to_excel(writer, sheet_name='calibrate')
                score_df.to_excel(writer, sheet_name='score')
        else:
            with pd.ExcelWriter(file, engine='openpyxl') as writer:
                book = load_workbook(file)
                writer.book = book
                df.to_excel(writer, sheet_name='calibrate')
                score_df.to_excel(writer, sheet_name='score')
        return self

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:15:01 2021

@author: linjianing
"""


import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from copy import deepcopy
from collections import OrderedDict
import autolrscorecard.variable_types.variable as vtype
from autolrscorecard.performance.withflag import gen_gaintable
from autolrscorecard.plotfig import plotROCKS, plotlift

getcontext().rounding = 'ROUND_HALF_UP'


class Entity:
    """实体."""

    def __init__(self, id, df, target=None, variable_options=None):
        self.id = id
        self.data = {'df': df, 'vops': variable_options, 'target': target,
                     'independents': df.columns.difference([target]).tolist(),
                     'proba': None}
        self.target = target
        self._create_variables(variable_options)
        self.pipe_X = self.df.loc[:, self.independents].copy(deep=True)
        if target:
            self.pipe_y = self.df.loc[:, self.target].copy(deep=True)
        self.steps = OrderedDict({})

    def update_data(self):
        comm = self.pipe_X.columns
        self.variable_options = {
            k: v for k, v in self.variable_options.items() if k in comm}
        return self

    def _create_variables(self, variable_options):
        variable_options = variable_options or {}
        ops = {'variable_type': pd.CategoricalDtype(), 'variable_shape': 'IDU', 'I_min': 3, 'U_min': 4}
        _vars = self.df.columns.difference([self.target])
        _var_types = self.df.loc[:, _vars].dtypes
        _ops = {k: {'variable_type': ops.get('variable_type') if not issubclass(v.type, np.number) else
                    vtype.Summ(1) if issubclass(v.type, np.floating) else vtype.Count(),
                    'variable_shape': 'D' if not issubclass(v.type, (np.number, vtype.Summ, vtype.Count, vtype.Ordinal))
                    else ops.get('variable_shape'),
                    'I_min': 2 if not issubclass(v.type, np.number) else ops.get('I_min'),
                    'U_min': 3 if not issubclass(v.type, np.number) else ops.get('U_min')}
                for k, v in _var_types.items()}
        for k, v in _ops.items():
            v.update(variable_options.get(k, {}))
        self.variable_options = _ops
        variable_types = {k: v.get('variable_type') for k, v in self.variable_options.items()}
        self.df = convert_all_variable_data(self.df, variable_types)

    @property
    def df(self):
        """Dataframe providing the data for the entity."""
        return self.data['df'].copy(deep=True)

    @df.setter
    def df(self, t_df):
        self.data['df'] = t_df

    @property
    def woe_df(self):
        """WOE Dataframe."""
        return self.data.get('woe_df', pd.DataFrame()).copy(deep=True)

    @woe_df.setter
    def woe_df(self, t_df):
        self.data['woe_df'] = t_df

    @property
    def target(self):
        """Target for the entity."""
        return self.data['target']

    @target.setter
    def target(self, tg):
        self.data['target'] = tg

    @property
    def proba(self):
        """Target for the entity."""
        return self.data['proba']

    @proba.setter
    def proba(self, tg):
        self.data['proba'] = tg

    @property
    def independents(self):
        """Independent for the entity."""
        return self.data['independents']

    @independents.setter
    def independents(self, indeps):
        self.data['independents'] = indeps

    @property
    def variable_options(self):
        """Variable_options for the entity."""
        return self.data['vops']

    @variable_options.setter
    def variable_options(self, vops):
        self.data['vops'] = vops

    def subentity(self, sub_id, cols):
        obj = deepcopy(self)
        obj.keep_variables(cols)
        obj.update_data()
        obj.id = sub_id
        return obj

    def drop_variables(self, variables):
        """删除变量."""
        cols = self.pipe_X.columns.intersection(list(variables))
        self.pipe_X = self.pipe_X.drop(cols, axis=1)
        return self

    def keep_variables(self, variables):
        """保留变量."""
        cols = self.pipe_X.columns.intersection(list(variables))
        self.pipe_X = self.pipe_X.loc[:, cols]
        return self

    @property
    def gain_table(self):
        """gain表."""
        return gen_gaintable(self.pred_y, self.pipe_y, bins=20, prob=True, output=False)

    def performance(self):
        """效果."""
        plotROCKS(self.pipe_y, self.pred_y, ks_label='{} KS&AUC'.format(self.id))
        gain_table = self.gain_table
        plotlift(gain_table, title='{} Lift'.format(self.id))
        return self

def convert_all_variable_data(df, vartypes):
    """变量转换."""
    df = df.copy(deep=True)
    for k, v in vartypes.items():
        if issubclass(type(v), (vtype.Summ, np.floating)):
            prec = v.prec
            str_prec = '0.'+'0'*prec
            df.loc[:, k] = pd.to_numeric(df.loc[:, k], errors='coerce', downcast='float').map(
                lambda x: float(Decimal(x).quantize(Decimal(str_prec))))
        elif issubclass(type(v), (vtype.Count, np.integer)):
            df.loc[:, k] = pd.to_numeric(df.loc[:, k], errors='coerce', downcast='integer')
        elif issubclass(type(v), (vtype.Category, vtype.Ordinal)):
            df.loc[:, k] = df.loc[:, k].astype(v.vtype)
        elif issubclass(type(v), pd.CategoricalDtype):
            df.loc[:, k] = df.loc[:, k].astype(v)
    return df

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:07:19 2021

@author: linjianing
"""


import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from .stepwise import stepwise_selection


class LassoLRCV:
    """lasso交叉验证逻辑回归."""

    def __init__(self, variable_options={}):
        pass

    def fit(self, X, y, verbose=False):
        """训练."""
        X_df = X.copy(deep=True)
        y_ser = y.copy(deep=True)
        X_names = X_df.columns.to_list()
        params = {'C': 1/np.logspace(np.log(1e-6), np.log(1), 50, base=math.e)}
        lass_lr = LogisticRegression(penalty='l1', solver='liblinear')
        while True:
            gscv = GridSearchCV(lass_lr, params)
            gscv.fit(X_df, y_ser)
            if not any(gscv.best_estimator_.coef_.ravel() < 0):
                break
            X_names = [k for k, v in dict(zip(X_names, gscv.best_estimator_.coef_.ravel())).items() if v > 0]
            if verbose:
                print(X_names)
            X_df = X_df.loc[:, X_names]
        coef_dict = dict(zip(X_names, gscv.best_estimator_.coef_.ravel()))
        self.lasso_vars = [k for k, v in coef_dict.items() if v > 0]
        return self

    def transform(self, X):
        """应用."""
        return X.loc[:, self.lasso_vars].copy(deep=True)

    def output(self):
        """输出报告."""
        pass


class Stepwise:
    """逐步回归筛选."""

    def __init__(self, variable_options={}, threshold_in=0.01, threshold_out=0.05):
        self.threshold_in = threshold_in
        self.threshold_out = threshold_out

    def fit(self, X, y, verbose=True):
        """训练."""
        X_df = X.copy(deep=True)
        step_out = stepwise_selection(X_df, y, threshold_in=self.threshold_in,
                                      threshold_out=self.threshold_out, verbose=verbose)
        self.stepwise_vars = step_out
        return self

    def transform(self, X):
        """应用."""
        return X.loc[:, self.stepwise_vars].copy(deep=True)

    def output(self):
        """输出报告."""
        pass


class LRCust:
    """逻辑回归."""

    def __init__(self, variable_options={}):
        pass

    def fit(self, X, y):
        """训练."""
        X_df = X.copy(deep=True)
        y_ser = y.copy(deep=True)
        lr = sm.Logit(y_ser, sm.add_constant(X_df)).fit()
        self.LRmodel_ = lr
        self.LRCust_vars = lr.summary2().tables[1].loc[:, 'Coef.'].index.difference(['const'])
        return self

    def transform(self, X):
        """应用."""
        return X.copy(deep=True).loc[:, self.LRCust_vars]

    def predict(self, X):
        """应用."""
        X_df = X.copy(deep=True).loc[:, self.LRCust_vars]
        lr = self.LRmodel_
        trns = lr.predict(sm.add_constant(X_df))
        trns.name = 'proba'
        return trns

    def output(self):
        """输出报告."""
        return self.LRmodel_.summary2().tables[1]

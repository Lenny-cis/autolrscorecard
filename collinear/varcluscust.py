# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:05:01 2021

@author: linjianing
"""


import os
import pandas as pd
from openpyxl import load_workbook
from varclushi import VarClusHi


def var_cluster(df, maxeigval2=1, maxclus=None, n_rs=0, feat_list=None,
                speedup=False):
    """
    处理多重共线性问题.

    使用varclus的方式进行变量聚类
    """
    vc = VarClusHi(df, feat_list, maxeigval2, maxclus, n_rs)
    vc.varclus(speedup)
    vc_rs = vc.rsquare
    cls_fst_var = vc_rs.sort_values(by=['RS_Ratio']).groupby(['Cluster'])\
        .head(1).loc[:, 'Variable']
    return (cls_fst_var.to_list(),
            vc.info.to_dict(orient='index'),
            vc_rs.to_dict(orient='index'))


class VarClusCust:
    """处理多重共线性问题."""

    def __init__(self, entity):
        self.entity = entity
        self.cluster_info = {}
        self.cluster_vars = {}
        self.cluster_rsquare = {}

    def fit(self, maxeigval2=1, maxclus=None, n_rs=0, speedup=False):
        """训练."""
        entity = self.entity
        X = entity.pipe_X
        cls_vars, vc_info, vc_rs = var_cluster(
            X, maxeigval2, maxclus, n_rs, None, speedup)
        self.cluster_info = vc_info
        self.cluster_vars = cls_vars
        self.cluster_rsquare = vc_rs
        entity.steps.update({
            'varclus': {'cluster_info': self.cluster_info,
                        'cluster_vars': self.cluster_vars,
                        'cluster_rsquare': self.cluster_rsquare}})
        return self

    def transform(self, entity=None):
        """应用."""
        if entity is None:
            entity = self.entity
        drop_vars = entity.pipe_X.columns.difference(self.cluster_vars)
        entity.pipe_X = entity.pipe_X.drop(drop_vars, axis=1)
        entity.update_data()
        return self

    def output(self):
        """输出报告."""
        pass

    def report(self, file):
        df = pd.DataFrame.from_dict(self.cluster_rsquare, orient='index')
        if not os.path.exists(file):
            with pd.ExcelWriter(file) as writer:
                df.to_excel(writer, sheet_name='varclus')
        else:
            with pd.ExcelWriter(file, engine='openpyxl') as writer:
                book = load_workbook(file)
                writer.book = book
                df.to_excel(writer, sheet_name='varclus')
        return self
# class VarClusCust:
#     """处理多重共线性问题."""

#     def __init__(self, variable_options={}, maxeigval2=1, maxclus=None, n_rs=0):
#         self.cluster_info = {}
#         self.cluster_vars = {}
#         self.cluster_rsquare = {}
#         self.maxeigval2 = maxeigval2
#         self.maxclus = maxclus
#         self.n_rs = n_rs

#     def fit(self, X, y, speedup=False):
#         """训练."""
#         cls_vars, vc_info, vc_rs = var_cluster(
#             X, self.maxeigval2, self.maxclus, self.n_rs, None, speedup)
#         self.cluster_info = vc_info
#         self.cluster_vars = cls_vars
#         self.cluster_rsquare = vc_rs
#         return self

#     def transform(self, X):
#         """应用."""
#         return X.loc[:, self.cluster_vars].copy(deep=True)

#     def output(self):
#         """输出报告."""
#         pass

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:05:01 2021

@author: linjianing
"""


from varclushi import VarClusHi


def var_cluster(df, maxeigval2=1, maxclus=None, n_rs=0, feat_list=None, speedup=False):
    """
    处理多重共线性问题.

    使用varclus的方式进行变量聚类
    """
    vc = VarClusHi(df, feat_list, maxeigval2, maxclus, n_rs)
    vc.varclus(speedup)
    vc_rs = vc.rsquare
    cls_fst_var = vc_rs.sort_values(by=['RS_Ratio']).groupby(['Cluster']).head(1).loc[:, 'Variable']
    return cls_fst_var, vc.info, vc_rs


class VarClusCust:
    """处理多重共线性问题."""

    def __init__(self, variable_options={}, maxeigval2=1, maxclus=None, n_rs=0):
        self.cluster_info = {}
        self.cluster_vars = {}
        self.cluster_rsquare = {}
        self.maxeigval2 = maxeigval2
        self.maxclus = maxclus
        self.n_rs = n_rs

    def fit(self, X, y, speedup=False):
        """训练."""
        cls_vars, vc_info, vc_rs = var_cluster(
            X, self.maxeigval2, self.maxclus, self.n_rs, None, speedup)
        self.cluster_info = vc_info
        self.cluster_vars = cls_vars
        self.cluster_rsquare = vc_rs
        return self

    def transform(self, X):
        """应用."""
        return X.loc[:, self.cluster_vars].copy(deep=True)

    def output(self):
        """输出报告."""
        pass

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:25:32 2021

@author: linjianing
"""


import os
import pandas as pd
from copy import deepcopy
from itertools import product
from autolrscorecard.entityset.entity import Entity
from autolrscorecard.woebin.woebinning import ExploreWOEBinning, WOEBinning
from autolrscorecard.collinear.varcluscust import VarClusCust
from autolrscorecard.model.modelestimator import LassoLRCV, LRCust, Stepwise
from autolrscorecard.plotfig.plotfig import (
    plot_bin, plot_repeat_split_performance)
from autolrscorecard.performance.modelstability import (
    repeatkfold_performance, vars_bin_psi, score_psi)


class EntitySet:
    """实体集合."""

    def __init__(self, id, entities=None):
        """创建实体集合.

        Example_:
            entities = {'train_sample': (train_df, {'target': 'flag, 'variable_options':{}})}.
        """
        self.id = id
        self.entity_dict = {}
        entities = entities or {}
        for entity in entities:
            df = entities[entity][0]
            kw = {}
            if len(entities[entity]) == 2:
                kw = entities[entity][1]
            self.entity_from_dataframe(entity_id=entity,
                                       dataframe=df,
                                       **kw)
        self.pipe_result = {}
        self.steps = {}
        self.best_bins = {}
        self.in_model_vars = {}

    def entity_from_dataframe(self, entity_id, dataframe, target=None,
                              variable_options=None):
        """从dataframe生成实体."""
        variable_options = variable_options or {}
        entity = Entity(entity_id,
                        dataframe,
                        target,
                        variable_options)
        self.entity_dict[entity.id] = entity
        return self

    def drop_entities(self, drop_list):
        """删除实体."""
        self.entity_dict = {k: v for k, v in self.entity_dict.items()
                            if k not in drop_list}
        return self

    @property
    def entities(self):
        """获取实体集合."""
        return list(self.entity_dict.values())

    def get_entity(self, entity_id):
        """获取实体."""
        return self.entity_dict[entity_id]

    def init_pipe_X(self, entity_id):
        """初始化pipe_X."""
        entity = self.get_entity(entity_id)
        entity.pipe_X = entity.df.loc[:, entity.variable_options.keys()]\
            .copy(deep=True)
        return self

    def explore_binning(self, entity_id, verbose=True,
                        save_file=None, **kwargs):
        """探索性分箱."""
        entity = self.get_entity(entity_id)
        entity_steps = self.steps.get(entity_id, {})
        ewb = ExploreWOEBinning(variable_options=entity.variable_options,
                                verbose=verbose, **kwargs)
        ewb.fit(entity.pipe_X, entity.pipe_y)
        entity_steps.update({'explore_binning': ewb})
        self.steps.update({entity_id: entity_steps})
        if save_file is not None:
            ewb.dump(save_file)
        return self

    def load_explore_bin(self, entity_id, file):
        """载入探索结果."""
        entity_steps = self.steps.get(entity_id, {})
        ewb = ExploreWOEBinning.load(file)
        entity_steps.update({'explore_binning': ewb})
        self.steps.update({entity_id: entity_steps})
        return self

    def explore_bin_search_best(self, entity_id, search_params={},
                                verbose=True, search_vars=None, **kwargs):
        """搜索最优分箱."""
        entity_steps = self.steps.get(entity_id, {})
        ewb = entity_steps['explore_binning']
        ewb.grid_search_best(search_params=search_params, verbose=verbose,
                             search_vars=search_vars, **kwargs)
        ewb.plot_best(plot_vars=search_vars)
        return self

    def WOE_binning(self, entity_id, variable_cuts, verbose=True,
                    **kwargs):
        """分箱."""
        entity = self.get_entity(entity_id)
        entity_steps = self.steps.get(entity_id, {})
        vop = {key: {
            'subjective_cut': cut,
            **entity.variable_options[key]}
            for key, cut in variable_cuts.items()}
        wb = WOEBinning(variable_options=vop, verbose=verbose, **kwargs)
        wb.fit(entity.pipe_X.loc[:, variable_cuts.keys()], entity.pipe_y)
        entity_steps.update({'WOE_binning': wb})
        self.steps.update({entity_id: entity_steps})
        entity.pipe_X = wb.transform(entity.pipe_X)
        return self

    def var_cluster(self, entity_id, maxeigval2=1,
                    maxclus=None, n_rs=0, speedup=True):
        """特征分群."""
        entity = self.get_entity(entity_id)
        entity_steps = self.steps.get(entity_id, {})
        vc = VarClusCust(variable_options=entity.variable_options,
                         maxeigval2=maxeigval2, maxclus=maxclus, n_rs=n_rs)
        vc.fit(entity.pipe_X, entity.pipe_y, speedup=speedup)
        entity_steps.update({'var_cluster': vc})
        self.steps.update({entity_id: entity_steps})
        entity.pipe_X = vc.transform(entity.pipe_X)
        return self

    def lasso_LRCV(self, entity_id):
        """lasso交叉验证逻辑回归."""
        entity = self.get_entity(entity_id)
        entity_steps = self.steps.get(entity_id, {})
        lslrcv = LassoLRCV(variable_options=entity.variable_options)
        lslrcv.fit(entity.pipe_X, entity.pipe_y)
        entity_steps.update({'lasso_LRCV': lslrcv})
        self.steps.update({entity_id: entity_steps})
        entity.pipe_X = lslrcv.transform(entity.pipe_X)
        return self

    def stepwise_LR(self, entity_id, threshold_in=0.1, threshold_out=0.1):
        """逐步回归."""
        entity = self.get_entity(entity_id)
        entity_steps = self.steps.get(entity_id, {})
        swlr = Stepwise(threshold_in=threshold_in, threshold_out=threshold_out)
        swlr.fit(entity.pipe_X, entity.pipe_y, verbose=False)
        entity_steps.update({'stepwise_LR': swlr})
        self.steps.update({entity_id: entity_steps})
        entity.pipe_X = swlr.transform(entity.pipe_X)
        return self

    def final_LR(self, entity_id):
        """lasso交叉验证逻辑回归."""
        entity = self.get_entity(entity_id)
        entity_steps = self.steps.get(entity_id, {})
        fnlr = LRCust(variable_options=entity.variable_options)
        fnlr.fit(entity.pipe_X, entity.pipe_y)
        entity_steps.update({'final_LR': fnlr})
        self.steps.update({entity_id: entity_steps})
        entity.pipe_X = fnlr.transform(entity.pipe_X)
        entity.pred_y = fnlr.predict(entity.pipe_X)
        rep = entity_steps['WOE_binning'].output()
        self.in_model_vars[entity_id] = rep.loc[
            rep.loc[:, 'var'].isin(list(entity.pipe_X.columns)), :]
        return self

    def batch_lr(self, entity_id):
        entity = self.get_entity(entity_id)
        entity_steps = self.steps.get(entity_id, {})
        if entity_steps == {}:
            return self
        entity_steps = entity_steps['explore_binning'].bin_dic
        tl = []
        kl = []
        for key, val in entity_steps.items():
            kl.append(key)
            ttl = []
            for vkey, vval in val.best_dic.items():
                ttl.append(vval['cut'])
            tl.append(ttl)
        tl = tuple(tl)
        bvl = product(*tl)
        n = 0
        for bv in bvl:
            pipe_X = entity.pipe_X
            pipe_y = entity.pipe_y
            bvd = dict(zip(kl, bv))
            vop = {key: {
                'subjective_cut': cut,
                **entity.variable_options[key]}
                for key, cut in bvd.items()}
            wb = WOEBinning(variable_options=vop, verbose=False)
            wb.fit(pipe_X.loc[:, bvd.keys()], pipe_y)
            pipe_X = wb.transform(entity.pipe_X)
            vc = VarClusCust(variable_options=entity.variable_options,
                             maxeigval2=0.5, maxclus=None, n_rs=0)
            vc.fit(pipe_X, pipe_y, speedup=True)
            pipe_X = vc.transform(pipe_X)
            lslrcv = LassoLRCV(variable_options=entity.variable_options)
            lslrcv.fit(pipe_X, pipe_y)
            pipe_X = lslrcv.transform(pipe_X)
            swlr = Stepwise(threshold_in=0.1, threshold_out=0.1)
            swlr.fit(pipe_X, pipe_y, verbose=False)
            pipe_X = swlr.transform(pipe_X)
            fnlr = LRCust(variable_options=entity.variable_options)
            fnlr.fit(pipe_X, pipe_y)
            pipe_X = fnlr.transform(pipe_X)
            pred_y = fnlr.predict(pipe_X)
            n += 1
            if n >= 10:
                break
        return bvl

    def transform(self, entity_id, X):
        """流式应用."""
        sX = X.copy(deep=True)
        entity_steps = self.steps[entity_id]
        for step, est in entity_steps.items():
            sX = est.transform(sX)
        return sX

    def predict(self, entity_id, X):
        """流式预测."""
        sX = self.transform(entity_id, X)
        entity_steps = self.steps[entity_id]
        est = entity_steps[list(entity_steps.keys())[-1]]
        px = est.predict(sX)
        return px

# =============================================================================
#     def pipe_fit(self, entity_id, estimators):
#         """流式训练."""
#         entity = self.get_entity(entity_id)
#         for (est_name, estimator) in estimators.items():
#             est = deepcopy(estimator)
#             setattr(est, 'variable_options', entity.variable_options)
#             est.fit(entity.pipe_X, entity.pipe_y)
#             self.steps[est_name] = est
#             entity.pipe_X = est.transform(entity.pipe_X)
#             if hasattr(est, 'best_bins'):
#                 self.best_bins = est.best_bins
#                 rep = est.output()
#         entity.pred_y = est.predict(entity.pipe_X)
#         self.in_model_vars = rep.loc[rep.loc[:, 'var'].isin(list(entity.pipe_X.columns)), :]
#         return self
#
#     def pipe_transform(self, X):
#         """流式应用."""
#         sX = X.copy(deep=True)
#         for step, est in self.steps.items():
#             sX = est.transform(sX)
#         return sX
#
#     def pipe_predict(self, X):
#         """流式预测."""
#         sX = self.pipe_transform(X)
#         est = self.steps[list(self.steps.keys())[-1]]
#         px = est.predict(sX)
#         return px
#
#     def performance(self, entity_id, n_r=10, n_s=5):
#         """效果."""
#         entity = self.get_entity(entity_id)
#         plot_bin(self.in_model_vars)
#         psi_df = repeatkfold_performance(entity.pipe_X, vars_bin_psi, n_r=n_r, n_s=n_s)
#         plot_repeat_split_performance(psi_df, 'VAR PSI', self.in_model_vars)
#         psi_df = repeatkfold_performance(pd.DataFrame(entity.pred_y), score_psi, n_r=n_r, n_s=n_s)
#         plot_repeat_split_performance(psi_df, 'SCORE PSI', pd.DataFrame({'describe': ['分数'], 'var': 'score'}))
#         entity.performance()
#         return self
# =============================================================================

    def performance(self, entity_id, n_r=10, n_s=5):
        """效果."""
        entity = self.get_entity(entity_id)
        in_model_vars = self.in_model_vars[entity_id]
        plot_bin(in_model_vars)
        psi_df = repeatkfold_performance(
            entity.pipe_X, vars_bin_psi, n_r=n_r, n_s=n_s)
        plot_repeat_split_performance(psi_df, 'VAR PSI', in_model_vars)
        psi_df = repeatkfold_performance(
            pd.DataFrame(entity.pred_y), score_psi, n_r=n_r, n_s=n_s)
        plot_repeat_split_performance(
            psi_df, 'SCORE PSI',
            pd.DataFrame({'describe': ['分数'], 'var': 'score'}))
        entity.performance()
        return self

    def output(self, entity_id, save_path):
        """输出结果."""
        filename = '_'.join([
            entity_id, 'model_report',
            pd.Timestamp.now().date().strftime('%Y%m%d')]) + '.xlsx'
        writer = pd.ExcelWriter(os.path.join(save_path, filename))
        in_model_vars = self.in_model_vars[entity_id]
        entity = self.get_entity(entity_id)
        entity_steps = self.steps[entity_id]
        for step, est in entity_steps.items():
            est_repor = est.output()
            if est_repor is not None:
                est_repor.to_excel(writer, step)
        gain_table = entity.gain_table
        in_model_vars.to_excel(writer, 'inModelVars')
        gain_table.to_excel(writer, 'gain_table')
        writer.save()
        writer.close()
        return self

    def component(self, entity_id):
        """组份."""
        entity = self.get_entity(entity_id)
        pipe_X = entity.pipe_X.copy(deep=True)
        raw_df = entity.df.copy(deep=True)
        pipe_X_cols = [x.rsplit('_', 2)[0] for x in list(pipe_X.columns)]
        raw_df = raw_df.loc[:, pipe_X_cols]
        ret_df = pd.concat([raw_df, pipe_X, entity.df.loc[:, [entity.target]],
                            pd.DataFrame(entity.pred_y)], axis=1)
        ret_df.loc[:, 'err_ratio'] = ret_df.loc[:, 'flag'] - ret_df.loc[:, 'proba']
        return ret_df

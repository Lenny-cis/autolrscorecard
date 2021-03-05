# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:25:32 2021

@author: linjianing
"""


import os
import pandas as pd
from copy import deepcopy
from .entity import Entity
from plotfig.plotfig import plot_bin, plot_repeat_split_performance
from performance.modelstability import repeatkfold_performance, vars_bin_psi, score_psi


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

    def entity_from_dataframe(self, entity_id, dataframe, target=None, variable_options=None):
        """从dataframe生成实体."""
        variable_options = variable_options or {}
        entity = Entity(entity_id,
                        dataframe,
                        target,
                        variable_options)
        self.entity_dict[entity.id] = entity
        return self

    def save_data(self, path):
        """储存数据为hdf文件."""
        sd = False
        for entity_id in self.entity_dict:
            entity_df = self.get_entity(entity_id).df
            if not sd:
                entity_df.to_hdf(path, entity_id, 'w')
                sd = True
            else:
                entity_df.to_hdf(path, entity_id, 'a')
        return self

    def drop_entities(self, drop_list):
        """删除实体."""
        self.entity_dict = {k: v for k, v in self.entity_dict.items() if k not in drop_list}
        return self

    def merge_entities(self, entity_ids, new_id, drop=False):
        """合并多个entity的df及variable_options.

        按列合并，最好不同的entity具有相同的index
        """
        assert isinstance(entity_ids, list), 'entity_ids must be list'
        new_entity = deepcopy(self.get_entity(entity_ids[0]))
        for entity_id in entity_ids[1:]:
            merged_entity = self.get_entity(entity_id)
            new_entity = new_entity.merge_entity(merged_entity)
        self.entity_dict[new_id] = new_entity
        if drop:
            drop_list = entity_ids[:]
            if new_id == entity_ids[0]:
                drop_list = entity_ids[1:]
            self.drop_entities(drop_list)
        return self

    @property
    def entities(self):
        """获取实体集合."""
        return list(self.entity_dict.values())

    def get_entity(self, entity_id):
        """获取实体."""
        return self.entity_dict[entity_id]

    def pipe_fit(self, entity_id, estimators):
        """流式训练."""
        entity = self.get_entity(entity_id)
        for (est_name, estimator) in estimators.items():
            est = deepcopy(estimator)
            setattr(est, 'variable_options', entity.variable_options)
            est.fit(entity.pipe_X, entity.pipe_y)
            self.steps[est_name] = est
            entity.pipe_X = est.transform(entity.pipe_X)
            if hasattr(est, 'best_bins'):
                self.best_bins = est.best_bins
                rep = est.output()
        entity.pred_y = est.predict(entity.pipe_X)
        self.in_model_vars = rep.loc[rep.loc[:, 'var'].isin(list(entity.pipe_X.columns)), :]
        return self

    def pipe_transform(self, X):
        """流式应用."""
        sX = X.copy(deep=True)
        for step, est in self.steps.items():
            sX = est.transform(sX)
        return sX

    def pipe_predict(self, X):
        """流式预测."""
        sX = self.pipe_transform(X)
        est = self.steps[list(self.steps.keys())[-1]]
        px = est.predict(sX)
        return px

    def performance(self, entity_id, n_r=10, n_s=5):
        """效果."""
        entity = self.get_entity(entity_id)
        plot_bin(self.in_model_vars)
        psi_df = repeatkfold_performance(entity.pipe_X, vars_bin_psi, n_r=n_r, n_s=n_s)
        plot_repeat_split_performance(psi_df, 'VAR PSI', self.in_model_vars)
        psi_df = repeatkfold_performance(pd.DataFrame(entity.pred_y), score_psi, n_r=n_r, n_s=n_s)
        plot_repeat_split_performance(psi_df, 'SCORE PSI', pd.DataFrame({'describe': ['分数'], 'var': 'score'}))
        entity.performance()
        return self

    def output(self, entity_id, save_path):
        """输出结果."""
        filename = '_'.join([entity_id, 'model_report', pd.Timestamp.now().date().strftime('%y%m%d')]) + '.xlsx'
        writer = pd.ExcelWriter(os.path.join(save_path, filename))
        entity = self.get_entity(entity_id)
        for step, est in self.steps.items():
            est_repor = est.output()
            if est_repor is not None:
                est_repor.to_excel(writer, step)
        gain_table = entity.gain_table
        self.performance(entity_id)
        self.in_model_vars.to_excel(writer, 'inModelVars')
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
        return ret_df

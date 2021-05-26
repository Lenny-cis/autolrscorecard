# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:40:28 2021

@author: linjianing
"""

import os
import pandas as pd
from openpyxl import load_workbook
from .withflag import gen_gaintable
from autolrscorecard.plotfig import plotKS, plotROC, plotlift, plotdist


class ModelPerformance:
    """模型性能展示."""
    def __init__(self, entity):
        self.entity = entity

    def plot(self):
        entity = self.entity
        y_ser = entity.df.loc[:, entity.target]
        proba = entity.proba
        plotROC(y_ser, proba, title=entity.id)
        plotKS(y_ser, proba, title=entity.id)
        plotdist(entity.pipe_X.loc[:, 'score'], title=entity.id)
        return self

    def plot_lift(self):
        entity = self.entity
        y_ser = entity.df.loc[:, entity.target]
        score = entity.pipe_X.loc[:, 'score']
        self.gain = gen_gaintable(score, y_ser, prob=False)
        plotlift(self.gain, entity.id)
        return self

    def report(self, file):
        self.plot()
        self.plot_lift()
        df = self.gain
        if not os.path.exists(file):
            with pd.ExcelWriter(file) as writer:
                df.to_excel(writer, sheet_name='gain')
        else:
            with pd.ExcelWriter(file, engine='openpyxl') as writer:
                book = load_workbook(file)
                writer.book = book
                df.to_excel(writer, sheet_name='gain')
        return self
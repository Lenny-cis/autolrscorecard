# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:17:44 2021

@author: linjianing
"""

import sys
import pickle as pkl
import pandas as pd
sys.path.append(r'F:\autolrscorecard')
sys.path.append(r'F:\lls\sme-pubinfoanalytics')
from variable_types import variable as vtype
from knowledgeModel.knowledgemodelfile import knowledgemodelfile
from entityset.entity import Entity
from entityset.entityset import EntitySet
from performance.performanceset import DescribeRatio, IVSet, KSAUCset
from woebin.woebinning import WOEBinning
from collinear.varcluscust import VarClusCust
from model.modelestimator import LassoLRCV, LRCust

sample = pd.read_hdf(r'E:\D99_DataBase\sme_knowledgemodel\features_and_sample_20210121.h5', 'sample')
features = pd.read_hdf(r'E:\D99_DataBase\sme_knowledgemodel\features_and_sample_20210121.h5', 'features')
features = features.loc[:, knowledgemodelfile.keys() & features.columns]

feature_options = {
    'LAST(holder_natural_year.SUM(holder.holdCapital))_growth_rate': {
        'variable_shape': 'D', 'thrd_n': 100, 'cut_cnt': 100},
    'COUNT(orggroup)_growth_rate': {'variable_shape': 'U', 'U_min': 3, 'thrd_n': 100},
    'COUNT(profession)': {'variable_shape': 'D', 'I_min': 2, 'thrd_n': 100},
    'LAST(business.AGE(foundDate))': {'variable_shape': 'U', 'thrd_n': 100},
    'MAX(related_business.AGE(foundDate) WHERE isRevoke = False)': {'variable_shape': 'U', 'thrd_n': 100},
    'TIME_SINCE_FIRST(related_change.regDate WHERE shallowrealholder_activite = True, unit=years)': {'variable_shape': 'D', 'thrd_n': 100},
    'TIME_SINCE_LAST(change.regDate WHERE importantItem = True, unit=years)': {'variable_shape': 'D', 'thrd_n': 100},
    'COUNT(related_business WHERE isRevoke = True)': {'variable_shape': 'I', 'I_min': 2, 'thrd_n': 50},
    'LAST(legalperson.AGE(BIRTH_DATE))': {'variable_shape': 'U', 'thrd_n': 100},
    'LAST(holder_natural_year.MAX(holder.holdScale))': {'variable_shape': 'U', 'thrd_n': 100},
    'SOFTCOUNT(negativity.regDate, punishSeverity)': {'variable_shape': 'I', 'I_min': 2, 'thrd_n': 100},
    'MODERATECOUNT(negativity.regDate, punishSeverity)': {'variable_shape': 'I', 'I_min': 2, 'thrd_n': 100},
    'SERIOUSCOUNT(negativity.regDate, punishSeverity)': {'variable_shape': 'I', 'I_min': 2, 'thrd_n': 100}}
feature_options = {ftkey: {**ftval, 'describe': knowledgemodelfile.get(ftkey)['describe']}
                   for ftkey, ftval in feature_options.items()}
feature_with_sample = features.merge(sample.loc[:, ['flag']], left_index=True, right_index=True)
es = Entity('lrm', feature_with_sample, 'flag', feature_options)

dr = DescribeRatio(feature_options, {'nonmissing': 0.9999, 'nonconcentration': 0.29, 'nunique': 10})
x_ns = feature_with_sample.columns.difference(['flag'])
dr.fit(feature_with_sample.loc[:, x_ns], feature_with_sample.loc[:, 'flag'])
# dr.describe_ratio
# dr.describe_vars

ivset = IVSet(es.variable_options, 0.01)
ivset.fit(feature_with_sample.loc[:, x_ns], feature_with_sample.loc[:, 'flag'])
ivset.ivset
ivset.iv_vars

ksset = KSAUCset(es.variable_options)
ksset.fit(feature_with_sample.loc[:, x_ns], feature_with_sample.loc[:, 'flag'])
ksset.ksaucset
ksset.ksauc_vars

woeb = WOEBinning(es.variable_options)
woeb.fit(feature_with_sample.loc[:, x_ns], feature_with_sample.loc[:, 'flag'], verbose=True)
woetrns = woeb.transform(feature_with_sample.loc[:, x_ns], keep=False)
test_best_bin = woeb.best_bins

vcc = VarClusCust(es.variable_options, maxeigval2=0.5)
vcc.fit(woetrns, feature_with_sample.loc[:, 'flag'])
test_vc = vcc.cluster_rsquare

lasso = LassoLRCV(es.variable_options)
lasso.fit(woetrns.loc[:, vcc.cluster_vars], feature_with_sample.loc[:, 'flag'])
woetrns = lasso.transform(woetrns.loc[:, vcc.cluster_vars])
lasso.lasso_vars

fnlr = LRCust(es.variable_options)
fnlr.fit(woetrns, feature_with_sample.loc[:, 'flag'])
fnlr.LRCust_vars

es = EntitySet('lrm')
es.entity_from_dataframe('train', feature_with_sample, 'flag', feature_options)
es.pipe_fit('train', {'woebin': WOEBinning(tolerance=0.05),
                      'varclus': VarClusCust(maxeigval2=0.5),
                      'lassocv': LassoLRCV(),
                      'finallr': LRCust()})
es.performance('train')
es.output('train', r'E:\A01_项目\A01_网查评分卡\AHP\05_AHP自动分箱')
txx = es.get_entity('train').pipe_X
pxx = es.pipe_predict(feature_with_sample)
lrt = es.steps['finallr'].LRmodel_.summary2().tables[1]
es.steps['woebin'].variable_options

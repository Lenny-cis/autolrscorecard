# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 14:33:17 2021

@author: linjianing
"""


import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from IPython import get_ipython
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from autolrscorecard.utils.performance_utils import (
    gen_ksseries, gen_cut, gen_cross)
from autolrscorecard.utils.woebin_utils import cut_to_interval

# matplotlib.use('agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
# get_ipython().run_line_magic('matplotlib', 'inline')


def plotROCKS(y, pred, ks_label='MODEL', pos_label=None):
    """画ROC曲线及KS曲线."""
    # 调整主次坐标轴
    xmajorLocator = matplotlib.ticker.MaxNLocator(6)
    xminorLocator = matplotlib.ticker.MaxNLocator(11)
    fpr, tpr, thrd = roc_curve(y, pred, pos_label=pos_label)
    ks_stp, w, ksTile = gen_ksseries(y, pred)
    auc_stp = auc(fpr, tpr)
    ks_x = fpr[w.argmax()]
    ks_y = tpr[w.argmax()]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    # 画ROC曲线
    ax[0].plot(fpr, tpr, 'r-', label='AUC=%.5f' % auc_stp, linewidth=0.5)
    ax[0].plot([0, 1], [0, 1], '-', color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax[0].plot([ks_x, ks_x], [ks_x, ks_y], 'r--', linewidth=0.5)
    ax[0].text(ks_x, (ks_x+ks_y)/2, '  KS=%.5f' % ks_stp)
    ax[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='FPR', ylabel='TPR',
              title='Receiver Operating Characteristic')
    ax[0].xaxis.set_major_locator(xmajorLocator)
    ax[0].xaxis.set_minor_locator(xminorLocator)
    ax[0].yaxis.set_minor_locator(xminorLocator)
    ax[0].fill_between(fpr, tpr, color='red', alpha=0.1)
    ax[0].legend()
    ax[0].grid(alpha=0.5, which='minor')
    # 画KS曲线
    ax[1].set_title('KS')
    allNum = len(y)
    eventNum = np.sum(y)
    nonEventNum = allNum - eventNum
    ks_p_x = (eventNum*ks_y + nonEventNum*ks_x)/allNum
    ax[1].plot(ksTile, w, 'r-', linewidth=0.5)
    ax[1].plot(ksTile, fpr, '-', color=(0.6, 0.6, 0.6),
               label='Good', linewidth=0.5)
    ax[1].text(ks_p_x, ks_y+0.05, 'Bad', color=(0.6, 0.6, 0.6))
    ax[1].plot(ksTile, tpr, '-', color=(0.6, 0.6, 0.6),
               label='Bad', linewidth=0.5)
    ax[1].text(ks_p_x, ks_x-0.05, 'Good', color=(0.6, 0.6, 0.6))
    ax[1].plot([ks_p_x, ks_p_x], [ks_stp, 0], 'r--', linewidth=0.5)
    ax[1].text(ks_p_x, ks_stp/2, '  KS=%.5f' % ks_stp)
    ax[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Prop', ylabel='TPR/FPR',
              title=ks_label+' KS')
    ax[1].xaxis.set_major_locator(xmajorLocator)
    ax[1].xaxis.set_minor_locator(xminorLocator)
    ax[1].yaxis.set_minor_locator(xminorLocator)
    ax[1].grid(alpha=0.5, which='minor')
    return fig


def plotROC(y, pred, title='MODEL', pos_label=None):
    """画ROC曲线及KS曲线."""
    # 调整主次坐标轴
    xmajorLocator = matplotlib.ticker.MaxNLocator(6)
    xminorLocator = matplotlib.ticker.MaxNLocator(11)
    fpr, tpr, thrd = roc_curve(y, pred, pos_label=pos_label)
    ks_stp, w, ksTile = gen_ksseries(y, pred)
    auc_stp = auc(fpr, tpr)
    ks_x = fpr[w.argmax()]
    ks_y = tpr[w.argmax()]
    fig, ax = plt.subplots()
    # 画ROC曲线
    ax.set_title(title + ' ROC')
    ax.plot(fpr, tpr, 'r-', label='AUC=%.5f' % auc_stp, linewidth=0.5)
    ax.plot([0, 1], [0, 1], '-', color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax.plot([ks_x, ks_x], [ks_x, ks_y], 'r--', linewidth=0.5)
    ax.text(ks_x, (ks_x+ks_y)/2, '  KS=%.5f' % ks_stp)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='FPR', ylabel='TPR')
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(xminorLocator)
    ax.fill_between(fpr, tpr, color='red', alpha=0.1)
    ax.legend()
    ax.grid(alpha=0.5, which='minor')
    return fig


def plotKS(y, pred, title='MODEL', pos_label=None):
    """画ROC曲线及KS曲线."""
    # 调整主次坐标轴
    xmajorLocator = matplotlib.ticker.MaxNLocator(6)
    xminorLocator = matplotlib.ticker.MaxNLocator(11)
    fpr, tpr, thrd = roc_curve(y, pred, pos_label=pos_label)
    ks_stp, w, ksTile = gen_ksseries(y, pred)
    ks_x = fpr[w.argmax()]
    ks_y = tpr[w.argmax()]
    fig, ax = plt.subplots()
    # 画KS曲线
    ax.set_title(title + ' KS')
    allNum = len(y)
    eventNum = np.sum(y)
    nonEventNum = allNum - eventNum
    ks_p_x = (eventNum*ks_y + nonEventNum*ks_x)/allNum
    ax.plot(ksTile, w, 'r-', linewidth=0.5)
    ax.plot(ksTile, fpr, '-', color=(0.6, 0.6, 0.6),
            label='Good', linewidth=0.5)
    ax.text(ks_p_x, ks_y+0.05, 'Bad', color=(0.6, 0.6, 0.6))
    ax.plot(ksTile, tpr, '-', color=(0.6, 0.6, 0.6),
            label='Bad', linewidth=0.5)
    ax.text(ks_p_x, ks_x-0.05, 'Good', color=(0.6, 0.6, 0.6))
    ax.plot([ks_p_x, ks_p_x], [ks_stp, 0], 'r--', linewidth=0.5)
    ax.text(ks_p_x, ks_stp/2, '  KS=%.5f' % ks_stp)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Prop', ylabel='TPR/FPR')
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(xminorLocator)
    ax.grid(alpha=0.5, which='minor')
    return fig


def plotlift(df, title):
    """画提升图."""
    f, ax = plt.subplots(figsize=(12, 9), tight_layout=True)
    ax2 = ax.twinx()
    f1 = ax.bar(range(len(df.index)), df['bad_num'])
    f2, = ax2.plot(range(len(df.index)), df['lift'], color='r')
    ax.set_xticks(list(range(len(df))))
    ax.set_xticklabels(df.loc[:, 'score_range'], rotation=45)
    ax.set_title(title)
    plt.legend([f1, f2], ['bad_num', 'lift'])
    plt.show()


def plotdist(score, title):
    """画分布图."""
    f, ax = plt.subplots(figsize=(12, 9), tight_layout=True)
    ax = sns.distplot(score, kde=True, bins=20)
    ax.set_title(title)
    plt.show()


def plot_bin(details):
    """画分箱图."""
    _ = plt.figure(tight_layout=True)
    for x in details.loc[:, 'var'].unique():
        detail = details.loc[details.loc[:, 'var'] == x, :]
        xticklabels = detail.loc[:, 'Bound']
        ax1 = sns.barplot(list(range(len(xticklabels))),
                          detail.loc[:, 'all_num'], label='Num')
        plt.xlabel(detail.loc[:, 'describe'].iloc[0])
        ax2 = ax1.twinx()
        ax2 = sns.lineplot(list(range(len(xticklabels))),
                           detail.loc[:, 'WOE'], color='r', label='WOE')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        hlist = h1 + h2
        llist = l1 + l2
        ax1.set_xticklabels(list(xticklabels))
        plt.legend(handles=hlist, labels=llist, loc='upper right')
        plt.show()
        plt.clf()
    plt.close('all')


def plot_repeat_split_performance(df, title, details):
    """画重复r折s次性能图."""
    _ = plt.figure(tight_layout=True)
    for x_name in df.columns:
        x_desc = details.loc[details.loc[:, 'var'] == x_name, 'describe']
        x = df.loc[:, x_name]
        sns.lineplot(y=x, x=x.index)
        plt.ylabel(None)
        plt.xlabel(x_desc.iloc[0])
        plt.title(title)
        plt.show()
        plt.clf()
    plt.close('all')


def plot_qcut_br(ser, y, var_type, n=20):
    """画等分图."""
    cut = gen_cut(ser, var_type, n=n, mthd='eqqt', precision=4)
    cross, cut = gen_cross(ser, y, cut, var_type)
    cross.loc[:, 'all_num'] = cross.sum(axis=1)
    cross.loc[:, 'event_prop'] = cross.loc[:, 1] / cross.loc[:, 'all_num']
    xticklabels = cut_to_interval(cut, var_type)
    _ = plt.figure(tight_layout=True)
    ax = sns.barplot(list(range(len(xticklabels))),
                     cross.loc[cross.index != -1, 'all_num'],
                     label='Num')
    ax2 = ax.twinx()
    ax2 = sns.lineplot(list(range(len(xticklabels))),
                       cross.loc[cross.index != -1, 'event_prop'],
                       label='Badrate')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    hlist = h1 + h2
    llist = l1 + l2
    ax.set_xticks(list(range(len(xticklabels))))
    ax.set_xticklabels(list(xticklabels.values()), rotation=45)
    plt.legend(handles=hlist, labels=llist, loc='upper right')
    plt.show()

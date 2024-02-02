import operator
from itertools import islice
from copy import copy
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy import stats

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

def get_jaccard_plot(df_importance, labels, f_name):
    df_importance = df_importance.fillna(-1)
    
    zero_data = np.zeros(shape=(len(df_importance), len(df_importance)))
    cols = labels
    d = pd.DataFrame(zero_data, columns=cols)

    data = []
    for i in range(len(df_importance)):
        aux = []
        for j in df_importance.columns:
            if df_importance[j].iloc[i] != -1:
                aux.append(j)
        data.append(aux)
        
    count = 0
    for i in range(len(d)):
        for k,j in enumerate(d.columns):
            d[j][i] = jaccard(data[i], data[k])
            
    
    my_cmap = copy(plt.cm.YlGnBu)
    my_cmap.set_over("white")
    my_cmap.set_under("white")
    
    fig, ax = plt.subplots(figsize=(8, 6),facecolor='white')
    g = sns.heatmap(d, vmin=0, fmt=".2f",ax=ax, cmap=my_cmap)
    l=list(np.arange(1,6+1))
    g.xaxis.get_label().set_fontsize(16)
    g.yaxis.get_label().set_fontsize(16)
    g.set_xticklabels(labels,rotation=90, fontsize=16)
    g.set_yticklabels(labels, rotation=0, fontsize=16)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    
    plt.savefig(f_name, bbox_inches='tight')
    plt.show()
    
    return d

def get_spearmans_plot(df_importance, labels, f_name):
    df_importance = df_importance.fillna(0)
    
    zero_data = np.zeros(shape=(len(df_importance), len(df_importance)))
    cols = labels
    d = pd.DataFrame(zero_data, columns=cols)

    data = []
    for i in range(len(df_importance)):
        data.append(df_importance.iloc[i])

    count = 0
    for i in range(len(d)):
        for k,j in enumerate(d.columns):
            d[j][i] = stats.spearmanr(data[i], data[k])[0]
            if stats.spearmanr(data[i], data[k])[1] > 0.05:
                d[j][i] = -1
            
    
    my_cmap = copy(plt.cm.YlGnBu)
    my_cmap.set_over("white")
    my_cmap.set_under("white")
    
    fig, ax = plt.subplots(figsize=(8, 6),facecolor='white')
    g = sns.heatmap(d, vmin=0.001, fmt=".2f",ax=ax, cmap=my_cmap)
    l=list(np.arange(1,6+1))
    g.xaxis.get_label().set_fontsize(16)
    g.yaxis.get_label().set_fontsize(16)
    g.set_xticklabels(labels,rotation=90, fontsize=16)
    g.set_yticklabels(labels, rotation=0, fontsize=16)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    
    plt.savefig(f_name, bbox_inches='tight')
    plt.show()
    
    return d
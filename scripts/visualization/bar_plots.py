import matplotlib.pyplot as plt
import operator
import pandas as pd
import numpy as np
import seaborn as sns 
import tensorflow as tf
import string
from .aux_dicts import *

def distributions_pass_fail(labels, Y, fname):
    fail_pass_total = []
    fail_pass = []
    number_students = []

    for c in range(np.max(np.unique(labels))+1):
        cluster = np.where(labels == c)[0]
        y_cluster = tf.gather(Y, list(cluster)).numpy()
        fail_pass.append((str(c), (y_cluster == 0).sum()/len(y_cluster)*100, (y_cluster == 1).sum()/len(y_cluster)*100))
        number_students.append((len(y_cluster), len(y_cluster)/len(labels)*100, (y_cluster == 0).sum()/len(y_cluster)*100))


    fail_pass = sorted(fail_pass, key=lambda tup: tup[1], reverse=True)
    number_students = sorted(number_students, key=lambda tup: tup[2], reverse=True)

    index = []
    for c in range(np.max(np.unique(labels))+1):
        print(string.ascii_uppercase[c], number_students[c])
        index.append(string.ascii_uppercase[c])
    index.append("Overall")
    
    fail_pass.append(("Overall", (Y == 0).sum()/len(Y)*100, (Y == 1).sum()/len(Y)*100))
    
    plt.bar(index, [i[1] for i in fail_pass], label="Pass", color='c')
    plt.bar(index, [i[2] for i in fail_pass], bottom=[i[1] for i in fail_pass], label="Fail", color='tomato')

    plt.ylabel('Percentage of students', fontsize=22)
    plt.yticks(fontsize=20)
    plt.xlabel('Clusters', fontsize=22)
    plt.xticks(index, rotation=0, ha='center', fontsize=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.38, 1.03), fontsize=22)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    
    plt.gca().spines['left'].set_linewidth(0.4)
    plt.gca().spines['bottom'].set_linewidth(0.4)
    
    plt.gca().spines['left'].set_color('grey')
    plt.gca().spines['bottom'].set_color('grey')

    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def percentage_students_per_feature_grouped_bar(labels, masks, feature_names, Y, fname):
    f_students = {}
    stds = {}

    # oder features for DSP 001
    order = ['Total Time Video', 'Content Alignment', 'Total Time Sessions', 'Total Clicks Video', 
            'Total Clicks Video Load', 'Std Time Between Sessions', 'Total Time Problem', 'Student Speed']
    for i in order:
        f_students[i] = []
        stds[i] = []
   
    overall = {}
    
    for c in range(np.max(np.unique(labels))+1):
        cluster = np.where(labels == c)[0]
        f_activated = tf.reduce_sum(masks, axis=0)

        for i in tf.where(f_activated)[:, 0]:
            a = np.where(masks[:, i] == 1)[0]
            a = list(set(a) & set(list(cluster)))
            l = np.zeros(np.abs(len(cluster)-len(a))).tolist()
            l.extend([1 for _ in range(len(a))])
            
            try:
                f_students[feature_names[i]].append(len(a)/len(cluster)*100)
                overall[feature_names[i]] += len(a)
            except:
                f_students[feature_names[i]] = []
                f_students[feature_names[i]].append(len(a)/len(cluster)*100)
                overall[feature_names[i]] = len(a)
        
    for f in f_students:
        f_students[f].append(overall[f]/len(labels)*100)


    df = pd.DataFrame(f_students)
    index = []
    for c in range(np.max(np.unique(labels))+1):
        index.append(string.ascii_uppercase[c])
    index.append('Overall')
    df['x'] = index

    df.plot(x='x', 
            kind='bar', 
            stacked=False, 
            figsize=(12,4),
            fontsize=22,
            width=0.65,
            color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'c', 'tab:olive', 'tab:pink'])
    plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1.03), fontsize=20)
    plt.xlabel('Clusters', fontsize=22)
    plt.xticks(rotation=0)
    plt.ylabel('Percentage of Students', fontsize=22)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    
    plt.gca().spines['left'].set_linewidth(0.4)
    plt.gca().spines['bottom'].set_linewidth(0.4)
    
    plt.gca().spines['left'].set_color('grey')
    plt.gca().spines['bottom'].set_color('grey')
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def percentage_students_per_feature_grouped_bar_shiffted(labels, masks, feature_names, Y, fname):
    f_students = {}
    stds = {}
    
    # oder features for DSP 001
    order = ['Total Time Video', 'Content Alignment', 'Total Time Sessions', 'Total Clicks Video', 
            'Total Clicks Video Load', 'Std Time Between Sessions', 'Total Time Problem', 'Student Speed']
    for i in order:
        f_students[i] = []
        stds[i] = []
        
    overall = {}
    
    for c in range(np.max(np.unique(labels))+1):
        cluster = np.where(labels == c)[0]
        f_activated = tf.reduce_sum(masks, axis=0)

        for i in tf.where(f_activated)[:, 0]:
            a = np.where(masks[:, i] == 1)[0]
            a = list(set(a) & set(list(cluster)))
            l = np.zeros(np.abs(len(cluster)-len(a))).tolist()
            l.extend([1 for _ in range(len(a))])
            
            f_students[feature_names[i]].append(len(a)/len(cluster)*100)
            try:
                overall[feature_names[i]] += len(a)
            except:
                overall[feature_names[i]] = len(a)
                
    for f in f_students:
        f_students[f].append(overall[f]/len(labels)*100)

    df = pd.DataFrame(f_students)
    index = []
    for c in range(np.max(np.unique(labels))+1):
        index.append(string.ascii_uppercase[c])
    index.append('Overall')
    df['index'] = index
    df = df.set_index(['index'])
    df = df.T

    df['x'] = order

    # Split each category name by spaces and join with a newline character
    df['x'] = ['\n'.join(c.split()) for c in df['x']]

    
    df.plot(x='x', 
            kind='bar', 
            stacked=False, 
            figsize=(12,4),
            fontsize=20,
            width=0.65,
            color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'c', 'tab:pink'])
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.03), fontsize=20)
    plt.xticks(rotation=0)
    plt.ylabel('Percentage of Students', fontsize=22)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    
    plt.gca().spines['left'].set_linewidth(0.4)
    plt.gca().spines['bottom'].set_linewidth(0.4)
    
    plt.gca().spines['left'].set_color('grey')
    plt.gca().spines['bottom'].set_color('grey')
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def feature_value_grouped_bar(labels, masks, feature_names, X, Y, fname):
    f_students = {}
    stds = {}

    # oder features for DSP 001
    order = ['Total Time Video', 'Content Alignment', 'Total Time Sessions', 'Total Clicks Video', 
            'Total Clicks Video Load', 'Std Time Between Sessions', 'Total Time Problem', 'Student Speed']
    for i in order:
        f_students[i] = []
        stds[i] = []
        
    for c in range(np.max(labels)+1):
        cluster = np.where(labels == c)[0]
        f_activated = tf.reduce_sum(masks, axis=0) # select only the activations in the current cluster

        for i in tf.where(f_activated)[:, 0]:
            a = np.where(masks[:, i] == 1)[0]
            a = list(set(a) & set(list(cluster)))
            
            f_students[feature_names[i]].append(tf.reduce_mean(X[a, :, i]).numpy())
            stds[feature_names[i]].append(tf.math.reduce_std(X[a, :, i]).numpy())

    # across all clusters
    for i in tf.where(f_activated)[:, 0]:
        f_students[feature_names[i]].append(tf.reduce_mean(X[:, :, i]).numpy())
        stds[feature_names[i]].append(tf.math.reduce_std(X[:, :, i]).numpy())
            
    df = pd.DataFrame(f_students)
    index = []
    for c in range(np.max(labels)+1):
        index.append('Cluster ' + str(c))
    index.append('Overall')
    df['x'] = index

    df.plot(x='x', 
            kind='bar', 
            stacked=False, 
            figsize=(15,5),
            fontsize=18,
            width=0.65,
            color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'c', 'tab:olive', 'tab:pink'])
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.03), fontsize=18)
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.ylabel('Feature Value Mean', fontsize=18)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    
    plt.gca().spines['left'].set_linewidth(0.4)
    plt.gca().spines['bottom'].set_linewidth(0.4)
    
    plt.gca().spines['left'].set_color('grey')
    plt.gca().spines['bottom'].set_color('grey')
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def feature_value_boxplot(labels, masks, feature_names, X, Y, fname):
    f_students = {}
    stds = {}

    # oder features for DSP 001
    order = ['Total Time Video', 'Content Alignment', 'Total Time Sessions', 'Total Clicks Video', 
            'Total Clicks Video Load', 'Std Time Between Sessions', 'Total Time Problem', 'Student Speed']
    for c in range(np.max(labels)+2):
        f_students[c] = {}
        for i in order:
            f_students[c][i] = []
        
    for c in range(np.max(labels)+1):
        cluster = np.where(labels == c)[0]
        f_activated = tf.reduce_sum(masks, axis=0)

        for i in tf.where(f_activated)[:, 0]:
            a = np.where(masks[:, i] == 1)[0]
            a = list(set(a) & set(list(cluster))) 
            f_students[c][feature_names[i]].append(tf.reduce_mean(X[a, :, i], axis=1).numpy())

    # across all clusters
    for i in tf.where(f_activated)[:, 0]:
        f_students[np.max(labels)+1][feature_names[i]].append(tf.reduce_mean(X[:, :, i], axis=1).numpy())
        
    data = []
    # dict_names = {  1: 'A',
    #                 0: 'B',
    #                 2: 'C'}
    for c in f_students:
        for f in f_students[c]:
            if c == np.max(labels)+1:
                data.append((f_students[c][f][0], 'Overall;'+f))
            else:
                # data.append((f_students[c][f][0], dict_names[c]+';'+f))
                data.append((f_students[c][f][0], string.ascii_uppercase[c]+';'+f))
                
    # Create the DataFrame
    df = pd.DataFrame(data, columns=['Feature Value', 'Config'])

    # Explode the 'Force' column to stack the NumPy arrays
    df = df.explode('Feature Value', ignore_index=True)
    
    df[['Group', 'Subgroup']] = df['Config'].str.split(pat=";", expand=True)
    
    # Create a box plot using Seaborn with custom spacing
    plt.figure(figsize=(15, 5))
    
    index = []
    for c in range(np.max(labels)+1):
        index.append(string.ascii_uppercase[c])
    index.append('Overall')

    ax = sns.boxplot(x='Group', order=index,
                     hue='Subgroup',
                     y='Feature Value', data=df,
                     palette=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'c', 'tab:olive', 'tab:pink'])
    ax.set_ylabel('Feature Value', fontsize=22)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20,  borderaxespad=0.)
    plt.tight_layout()
    plt.xlabel('Clusters', fontsize=22)
    plt.xticks(rotation=0, fontsize=20)
    plt.yticks(fontsize=20)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    
    plt.gca().spines['left'].set_linewidth(0.4)
    plt.gca().spines['bottom'].set_linewidth(0.4)
    plt.gca().spines['left'].set_color('grey')
    plt.gca().spines['bottom'].set_color('grey')
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def feature_value_boxplot_shiffted(labels, masks, feature_names, X, Y, fname):
    f_students = {}
    stds = {}

    # oder features for DSP 001
    order = ['Total Time Video', 'Content Alignment', 'Total Time Sessions', 'Total Clicks Video', 
            'Total Clicks Video Load', 'Std Time Between Sessions', 'Total Time Problem', 'Student Speed']
    for c in range(np.max(labels)+2):
        f_students[c] = {}
        for i in order:
            f_students[c]['\n'.join(i.split())] = []
        
    # for c in range(np.max(labels)+1):
    for c in range(np.max(labels)+1):
        cluster = np.where(labels == c)[0]
        f_activated = tf.reduce_sum(masks, axis=0)

        for i in tf.where(f_activated)[:, 0]:
            a = np.where(masks[:, i] == 1)[0]
            a = list(set(a) & set(list(cluster))) 
            f_students[c]['\n'.join(feature_names[i].split())].append(tf.reduce_mean(X[a, :, i], axis=1).numpy())

    # across all clusters
    for i in tf.where(f_activated)[:, 0]:
        f_students[np.max(labels)+1]['\n'.join(feature_names[i].split())].append(tf.reduce_mean(X[:, :, i], axis=1).numpy())
        
    data = []
    for c in f_students:
        for f in f_students[c]:
            if c == np.max(labels)+1:
                data.append((f_students[c][f][0], 'Overall;'+f))
            else:
                data.append((f_students[c][f][0], string.ascii_uppercase[c]+';'+f))
                
    # Create the DataFrame
    df = pd.DataFrame(data, columns=['Feature Value', 'Config'])

    # Explode the 'Force' column to stack the NumPy arrays
    df = df.explode('Feature Value', ignore_index=True)
    
    df[['Group', 'Subgroup']] = df['Config'].str.split(pat=";", expand=True)

    
    # Create a box plot using Seaborn with custom spacing
    plt.figure(figsize=(15, 5))
    
    index = ['\n'.join(c.split()) for c in order]
    
    ax = sns.boxplot(x='Subgroup', order=index,
                     hue='Group',
                     y='Feature Value', data=df,
                     palette=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'c', 'tab:pink'])
    ax.set_ylabel('Feature Value', fontsize=22)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=22,  borderaxespad=0.)
    plt.tight_layout()
    plt.xlabel('')
    plt.xticks(rotation=0, fontsize=20)
    plt.yticks(fontsize=20)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    
    plt.gca().spines['left'].set_linewidth(0.4)
    plt.gca().spines['bottom'].set_linewidth(0.4)
    plt.gca().spines['left'].set_color('grey')
    plt.gca().spines['bottom'].set_color('grey')
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def failing_distributions(labels, Y):
    fail_pass_total = []
    fail_pass = []
    for c in range(np.max(np.unique(labels))+1):
        cluster = np.where(labels == c)[0]
        y_cluster = tf.gather(Y, list(cluster)).numpy()

        fail_pass_total.append(("Cluster "+str(c), (y_cluster == 0).sum()/len(Y), (y_cluster == 1).sum()/len(Y)))
        fail_pass.append(("Cluster "+str(c), (y_cluster == 0).sum()/len(y_cluster), (y_cluster == 1).sum()/len(y_cluster)))
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # ax[0].bar([i[0] for i in fail_pass_total], [i[1] for i in fail_pass_total], label="Pass", color='c')
    ax[0].bar([i[0] for i in fail_pass_total], [i[2] for i in fail_pass_total], label="Fail", color='tomato')
    ax[0].set_title('Distribution of Pass-Fail')
    ax[0].set_ylabel('Percentage of total students')
    ax[0].set_xticklabels([i[0] for i in fail_pass_total], rotation=45, ha='right')
    ax[0].legend()
    
    # ax[1].bar([i[0] for i in fail_pass], [i[1] for i in fail_pass], label="Pass", color='c')
    ax[1].bar([i[0] for i in fail_pass], [i[2] for i in fail_pass], label="Fail", color='tomato')
    ax[1].set_title('Distribution of Pass-Fail')
    ax[1].set_ylabel('Percentage of students per cluster')
    ax[1].set_xticklabels([i[0] for i in fail_pass], rotation=45, ha='right')
    ax[1].legend()
    plt.show()

def number_students_per_feature(labels, masks, feature_names, Y):
    df = pd.DataFrame()
    clusters_length = []
    avg_feat_per_cluster = []

    for c in range(np.max(labels)+1):
        cluster = np.where(labels == c)[0]
        clusters_length.append(len(cluster))
        avg_feat = np.mean(tf.reduce_sum(tf.gather(masks, list(cluster)), axis=1))
        avg_feat_per_cluster.append(avg_feat)

        f_activated = tf.reduce_sum(masks, axis=0)
        f_students = []

        for i in tf.where(f_activated)[:, 0]:
            
            a = np.where(masks[:, i] == 1)[0]
            a = list(set(a) & set(list(cluster)))

            f_students.append((feature_names[i], (Y[a] == 0).sum(), (Y[a] == 1).sum()))

        plt.barh([i[0] for i in f_students], [i[1] for i in f_students], label="Pass", color='c')
        plt.barh([i[0] for i in f_students], [i[2] for i in f_students], left=[i[1] for i in f_students], label="Fail", color='tomato')
        plt.title('Cluster '+str(c))
        plt.xlabel('Number of students with feature activated')
        plt.legend()
        plt.show()

    df['Cluster'] = [i for i in range(np.max(labels)+1)]
    df['# instances'] = clusters_length
    df['Avg Features'] = avg_feat_per_cluster
    
    return df

def percentage_students_per_feature(labels, masks, feature_names, Y):
    for c in range(np.max(labels)+1):
        cluster = np.where(labels == c)[0]
        f_activated = tf.reduce_sum(masks, axis=0)

        f_students = []
        for i in tf.where(f_activated)[:, 0]:
            a = np.where(masks[:, i] == 1)[0]
            a = list(set(a) & set(list(cluster)))

            f_students.append((feature_names[i], [(Y[a] == 0).sum()/len(cluster), (Y[a] == 1).sum()/len(cluster)]))

        plt.barh([i[0] for i in f_students], [i[1][0] for i in f_students], label="Pass", color='c')
        plt.barh([i[0] for i in f_students], [i[1][1] for i in f_students], left=[i[1][0] for i in f_students], label="Fail", color='tomato')
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Percentage of students', fontsize=14)
        plt.legend(fontsize=14)
        plt.show()

def feature_value(labels, masks, feature_names, X, Y):
    f_students = {}
    xlim_max = 0
    for c in range(np.max(labels)+1):

        cluster = np.where(labels == c)[0]
        f_activated = tf.reduce_sum(tf.gather(masks, list(cluster)), axis=0)
        total_mean_max = 0
        for i in tf.where(f_activated)[:, 0]:
            if tf.reduce_mean(X[:, :, i]) > total_mean_max:
                total_mean_max = tf.reduce_mean(X[:, :, i])

        f_students[c] = []
        f_activated = tf.reduce_sum(masks, axis=0) # so we can keep the same axis for all plots

        for i in tf.where(f_activated)[:, 0]:
            a = np.where(masks[:, i] == 1)[0]
            a = list(set(a) & set(list(cluster))) # select only the activations in the current cluster
            
            target_0 = [j for j in a if Y[j] == 0]
            target_1 = [j for j in a if Y[j] == 1]
            
            if len(target_0) > 0 and len(target_1) > 0:
                f_students[c].append((feature_names[i], [tf.reduce_mean(X[target_0, :, i]), 
                                   tf.reduce_mean(X[target_1, :, i])]))
                if tf.reduce_mean(X[target_0, :, i])+tf.reduce_mean(X[target_1, :, i]) > xlim_max:
                    xlim_max = tf.reduce_mean(X[target_0, :, i])+tf.reduce_mean(X[target_1, :, i])
            elif len(target_0) == 0:
                f_students[c].append((feature_names[i], [0, 
                                   tf.reduce_mean(X[target_1, :, i])]))
                if tf.reduce_mean(X[target_1, :, i]) > xlim_max:
                    xlim_max = tf.reduce_mean(X[target_1, :, i])
            else:
                f_students[c].append((feature_names[i], [tf.reduce_mean(X[target_0, :, i]), 0]))
                if tf.reduce_mean(X[target_0, :, i]) > xlim_max:
                    xlim_max = tf.reduce_mean(X[target_0, :, i])

    for c in range(np.max(labels)+1):
        plt.barh([i[0] for i in f_students[c]], [i[1][0] for i in f_students[c]], label="Pass", color='c')
        plt.barh([i[0] for i in f_students[c]], [i[1][1] for i in f_students[c]], left=[i[1][0] for i in f_students[c]], label="Fail", color='tomato')
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Feature Value Mean', fontsize=14)
        plt.legend(fontsize=14)
        plt.xlim(0, xlim_max + 0.05)
        plt.show()

def relative_feature_value(labels, masks, feature_names, X, Y):
    df = pd.DataFrame()

    xlim_max = 0
    xlim_min = 100
    f_students = {}
    for c in range(np.max(labels)+1):
        cluster = np.where(labels == c)[0]
        f_activated = tf.reduce_sum(masks, axis=0)

        f_students[c] = []
        for i in tf.where(f_activated)[:, 0]:
            a = np.where(masks[:, i] == 1)[0]
            a = list(set(a) & set(list(cluster))) 

            target_0 = [j for j in a if Y[j] == 0]
            target_1 = [j for j in a if Y[j] == 1]
            
            if len(target_0) > 0 and len(target_1) > 0:
                f_students[c].append((feature_names[i], [tf.reduce_mean(X[target_0, :, i])-tf.reduce_mean(X[:, :, i]), 
                                   tf.reduce_mean(X[target_1, :, i])-tf.reduce_mean(X[:, :, i])]))
                x = tf.reduce_mean(X[target_0, :, i])-tf.reduce_mean(X[:, :, i])+tf.reduce_mean(X[target_1, :, i])-tf.reduce_mean(X[:, :, i])
                if x > xlim_max:
                    xlim_max = x
                if x < xlim_min:
                    xlim_min = x

            elif len(target_0) == 0:
                f_students[c].append((feature_names[i], [0, 
                                   tf.reduce_mean(X[target_1, :, i])-tf.reduce_mean(X[:, :, i])]))
                x = tf.reduce_mean(X[target_1, :, i])-tf.reduce_mean(X[:, :, i])
                if x > xlim_max:
                    xlim_max = x
                if x < xlim_min:
                    xlim_min = x
            else:
                f_students[c].append((feature_names[i], [tf.reduce_mean(X[target_0, :, i])-tf.reduce_mean(X[:, :, i]), 
                               0]))
                x = tf.reduce_mean(X[target_0, :, i])-tf.reduce_mean(X[:, :, i])
                if x > xlim_max:
                    xlim_max = x
                if x < xlim_min:
                    xlim_min = x

    for c in range(np.max(labels)+1):
        plt.barh([i[0] for i in f_students[c]], [i[1][0] for i in f_students[c]], label="Pass", color='c')
        plt.barh([i[0] for i in f_students[c]], [i[1][1] for i in f_students[c]], left=[i[1][0] for i in f_students[c]], label="Fail", color='tomato')
        plt.title('Cluster '+str(c))
        plt.xlabel('Relative Feature Value Mean')
        plt.legend()
        plt.xlim(xlim_min-0.05, xlim_max+0.05) # fix this
        plt.show()

    f_mean = []
    f_names = []
    f_max = []
    f_min = []
    
    for i in tf.where(f_activated)[:, 0]:
        f_mean.append(tf.reduce_mean(X[:, :, i]).numpy())
        f_max.append(tf.reduce_max(X[:, :, i]).numpy())
        f_names.append(feature_names[i])
        f_min.append(tf.reduce_min(X[:, :, i]).numpy())

    df['Feature'] = f_names
    df['Avg Value'] = f_mean
    df['Max Value'] = f_max
    df['Min Value'] = f_min
    return df

def percentage_courses_per_feature(df_importance):
    
    df_importance = df_importance.fillna(-1)
    
    feats = {}

    for i in range(len(df_importance)):
        aux = []
        for j in df_importance.columns:
            if df_importance[j].iloc[i] != -1:
                try: 
                    feats[feature_names_dict[j]] += 1
                except:
                    feats[feature_names_dict[j]] = 1

    feats = dict(sorted(feats.items(), key=operator.itemgetter(1), reverse=True))
    feats_fail = feats
    index = list(feats.keys()) 

    newList = [x / len(df_importance) * 100 for x in list(feats.values())]

    plt.figure(figsize=(12, 5))
    plt.bar(index, newList)

    plt.yticks(fontsize=18)
    plt.ylabel('Percentage of courses', fontsize=22)
    plt.xticks(index, rotation=90, ha='center', fontsize=18)
    plt.show()
    
    return index, feats
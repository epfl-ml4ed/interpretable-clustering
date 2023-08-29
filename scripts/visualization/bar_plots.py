import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

def distributions_pass_fail(labels, Y):
    fail_pass_total = []
    fail_pass = []
    for c in range(np.max(np.unique(labels))+1):
        cluster = np.where(labels == c)[0]
        y_cluster = tf.gather(Y, list(cluster)).numpy()

        fail_pass_total.append(("Cluster "+str(c), (y_cluster == 0).sum()/len(Y), (y_cluster == 1).sum()/len(Y)))
        fail_pass.append(("Cluster "+str(c), (y_cluster == 0).sum()/len(y_cluster), (y_cluster == 1).sum()/len(y_cluster)))
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].bar([i[0] for i in fail_pass_total], [i[1] for i in fail_pass_total], label="Pass", color='tomato')
    ax[0].bar([i[0] for i in fail_pass_total], [i[2] for i in fail_pass_total], bottom=[i[1] for i in fail_pass_total], label="Fail", color='c')
    ax[0].set_title('Distribution of Pass-Fail')
    ax[0].set_ylabel('Percentage of total students')
    ax[0].set_xticklabels([i[0] for i in fail_pass_total], rotation=45, ha='right')
    ax[0].legend()
    
    ax[1].bar([i[0] for i in fail_pass], [i[1] for i in fail_pass], label="Pass", color='tomato')
    ax[1].bar([i[0] for i in fail_pass], [i[2] for i in fail_pass], bottom=[i[1] for i in fail_pass], label="Fail", color='c')
    ax[1].set_title('Distribution of Pass-Fail')
    ax[1].set_ylabel('Percentage of students per cluster')
    ax[1].set_xticklabels([i[0] for i in fail_pass], rotation=45, ha='right')
    ax[1].legend()
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
    # ax[0].bar([i[0] for i in fail_pass_total], [i[1] for i in fail_pass_total], label="Pass", color='tomato')
    ax[0].bar([i[0] for i in fail_pass_total], [i[2] for i in fail_pass_total], label="Fail", color='c')
    ax[0].set_title('Distribution of Pass-Fail')
    ax[0].set_ylabel('Percentage of total students')
    ax[0].set_xticklabels([i[0] for i in fail_pass_total], rotation=45, ha='right')
    ax[0].legend()
    
    # ax[1].bar([i[0] for i in fail_pass], [i[1] for i in fail_pass], label="Pass", color='tomato')
    ax[1].bar([i[0] for i in fail_pass], [i[2] for i in fail_pass], label="Fail", color='c')
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

        f_activated = tf.reduce_sum(tf.gather(masks, list(cluster)), axis=0)

        f_students = []
        for i in tf.where(f_activated)[:, 0]:
            a = np.where(masks[:, i] == 1)[0]
            a = list(set(a) & set(list(cluster)))

            f_students.append((feature_names[i], (Y[a] == 0).sum(), (Y[a] == 1).sum()))

        plt.barh([i[0] for i in f_students], [i[1] for i in f_students], label="Pass", color='tomato')
        plt.barh([i[0] for i in f_students], [i[2] for i in f_students], left=[i[1] for i in f_students], label="Fail", color='c')
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
        f_activated = tf.reduce_sum(tf.gather(masks, list(cluster)), axis=0)

        f_students = []
        for i in tf.where(f_activated)[:, 0]:
            a = np.where(masks[:, i] == 1)[0]
            a = list(set(a) & set(list(cluster)))

            f_students.append((feature_names[i], [(Y[a] == 0).sum()/len(cluster), (Y[a] == 1).sum()/len(cluster)]))

        plt.barh([i[0] for i in f_students], [i[1][0] for i in f_students], label="Pass", color='tomato')
        plt.barh([i[0] for i in f_students], [i[1][1] for i in f_students], left=[i[1][0] for i in f_students], label="Fail", color='c')
        plt.title('Cluster '+str(c))
        plt.xlabel('Percentage of students with feature activated')
        plt.legend()
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
    
        for i in tf.where(f_activated)[:, 0]:
            a = np.where(masks[:, i] == 1)[0]
            a = list(set(a) & set(list(cluster))) 
            
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
        plt.barh([i[0] for i in f_students[c]], [i[1][0] for i in f_students[c]], label="Pass", color='tomato')
        plt.barh([i[0] for i in f_students[c]], [i[1][1] for i in f_students[c]], left=[i[1][0] for i in f_students[c]], label="Fail", color='c')
        plt.title('Cluster '+str(c))
        plt.xlabel('Feature Value Mean')
        plt.legend()
        plt.xlim(0, xlim_max + 0.05)
        plt.show()

def relative_feature_value(labels, masks, feature_names, X, Y):
    df = pd.DataFrame()

    xlim_max = 0
    xlim_min = 100
    f_students = {}
    for c in range(np.max(labels)+1):
        cluster = np.where(labels == c)[0]
        f_activated = tf.reduce_sum(tf.gather(masks, list(cluster)), axis=0)

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
        plt.barh([i[0] for i in f_students[c]], [i[1][0] for i in f_students[c]], label="Pass", color='tomato')
        plt.barh([i[0] for i in f_students[c]], [i[1][1] for i in f_students[c]], left=[i[1][0] for i in f_students[c]], label="Fail", color='c')
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
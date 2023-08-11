import sys
import os
import tensorflow as tf
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans, OPTICS, SpectralClustering
from sklearn.metrics import pairwise_distances
from tslearn.metrics import cdist_dtw
import numpy as np
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"/scripts")

from preprocessing import preprocess
from gating import *

def kmeans_(n):
    return KMeans(n_clusters=n, random_state=0)

def optics_(n):
    return OPTICS(min_samples=n, metric='euclidean')

def spectral_clustering(n):
    '''
        SpectralClustering to cluster similarities between instances
    '''
    return SpectralClustering(n_clusters=n, affinity='precomputed')

def euclidean_distance(X):
    return pairwise_distances(X, metric='euclidean')

def dtw_similatirity(X, window):
    return cdist_dtw(X, global_constraint="sakoe_chiba", sakoe_chiba_radius=window)

def compute_distance_matrix(X, metric):
    distance_sum = np.zeros((X.shape[0], X.shape[0]))
    for i in np.arange(X.shape[2]):
        if metric == 'euclidean':
            distance_matrix = euclidean_distance(X[:,:,i])
        elif metric == 'dtw':
            distance_matrix = dtw_similatirity(X[:,:,i], 2)
        if np.max(distance_matrix) > 0:
            D = distance_matrix/np.max(distance_matrix)
        else:
            D = distance_matrix  
        np.fill_diagonal(D, 0)
        distance_sum += D

    return distance_sum / X.shape[2]

def compute_number_clusters(data, model, metric, distance_matrix=[], minimization=False, verbose=True):
    '''
        minimization: if True the evaluated metric is a minimization problem 
    '''
    if minimization:
        best_score = sys.maxint
    else:
        best_score = 0
    best_n = 0
    best_labels = []
    for n in range(2, 8):
        labels = model(n).fit_predict(data)
        if distance_matrix != []:
            score = metric(distance_matrix, labels, metric='precomputed')
        else:
            score = metric(data, labels)
        if verbose:
            print((n, score))
        if minimization:
            if score < best_score:
                best_score = score
                best_n = n
                best_labels = labels
        else:
            if score > best_score:
                best_score = score
                best_n = n
                best_labels = labels

    return best_score, best_n, best_labels

def get_truncated_features(MODEL_PATH, filename, course, path, percentile, feature_types, metadata, hard_fail):
    x_train, x_test, x_val, y_train, y_test, y_val, feature_names, patterns = preprocess(course, path, percentile, feature_types, metadata, hard_fail)
    
    # Concat features & labels
    X = np.concatenate([x_train, x_val, x_test], axis=0)
    Y = np.concatenate([y_train, y_val, y_test], axis=0)
    P = np.concatenate(patterns, axis=0)
    
    # Set up parameters and model to train
    meta = {'gumbel_temp': 1, 'gumbel_noise': 1e-8}
    model = MaskingModel(n_groups=x_train.shape[-1])
    
    # Load model
    model.load_weights(MODEL_PATH + filename).expect_partial()

    # Get masks
    masks = model.get_mask(X, meta)
   
    # Reduce over feature
    f_activated = tf.reduce_sum(masks, axis=0)
    # f_activations = [(feature_names[i], f_activated[i].numpy()) for i in tf.where(f_activated)[:, 0]]
    
    # Truncate to activated features
    activated = tf.where(f_activated)[:, 0]
    masks_a = tf.gather(masks, activated, axis=-1)
    
    X_a = tf.gather(X, activated, axis=-1)
    X_a = X_a.numpy()
    
    # Expand masks to repeat for each week
    masks_expanded = tf.expand_dims(masks_a, axis=1)
    masks_expanded = tf.repeat(masks_expanded, repeats=[X_a.shape[1]], axis=1)
    # masks_expanded.shape
    X_masked = masks_expanded*X_a

    return feature_names, masks, X_masked, X, Y, P

def get_truncated_features_flatten(MODEL_PATH, filename, course, path, percentile, feature_types, metadata, hard_fail):
    feature_names, masks, X_masked, X, Y, P = get_truncated_features(MODEL_PATH, filename, course, path, percentile, feature_types, metadata, hard_fail)
    # Flatten to 2D 
    X_masked = tf.reshape(X_masked,[X_masked.shape[0], X_masked.shape[1]*X_masked.shape[2]])
    
    return feature_names, masks, X_masked, X, Y, P


def get_x_flatten(course, path, percentile, feature_types, metadata, hard_fail):
    x_train, x_test, x_val, y_train, y_test, y_val, feature_names , patterns = preprocess(course, path, percentile, feature_types, metadata, hard_fail)
    X = np.concatenate([x_train, x_val, x_test], axis=0)
    Y = np.concatenate([y_train, y_val, y_test], axis=0)
    P = np.concatenate(patterns, axis=0)
    
    X_flatten = tf.reshape(X,[X.shape[0], X.shape[1]*X.shape[2]])
    print("X_flatten shape: {0}".format(X_flatten.shape))
    return feature_names, X_flatten, X, Y, P
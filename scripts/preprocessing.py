#!/usr/bin/env python
# coding: utf-8

import ast
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler

TRAIN_SIZE = 0.8
TEST_SIZE = 0.1
VAL_SIZE = 0.1

WEEK_TYPE = 'eq_week'

def preprocess(course, path, percentile, feature_types, metadata, normalization='min-max'):
    ''' Pre-process data related to given course and perform train-val-test split

    Args:
        course: (str) Course name/identifier
        path: (str) Course data path
        percentile: (float) Percentile point between 0 and 1 for early prediction
        feature_types: (List[str]) Feature types
        metadata: (pd.DataFrame) Courses metadata

    Returns:
        Data split as x_train, x_test, x_val, y_train, y_test, y_val, feature_names, pattern_labels
    '''
    feature_list = []
    names_list = []
    total_weeks = list(metadata[metadata['course_id'] == course.replace('_', '-')]['weeks'])[0]
    num_weeks = int(np.round(total_weeks * percentile))

    hard_fails = pd.read_csv(f'{path}/eq_week-marras_et_al-{course}/feature_labels.csv', index_col=0)
    hard_fail_idx = hard_fails.index.tolist()
    labels = hard_fails['label-pass-fail'].values


    scaler = MinMaxScaler()

    for feature_type in feature_types:
        filepath = path + WEEK_TYPE + '-' + feature_type + '-' + course

        feature_current = np.load(filepath+'/feature_values.npz')['feature_values']
        feature_current = np.nan_to_num(feature_current, nan=0)
        feature_current = feature_current[:, :num_weeks, :]
        feature_current = np.pad(feature_current, pad_width=((0, 0), (0, num_weeks-feature_current.shape[1]), 
                                                             (0, 0)), mode='constant', constant_values=0)
        # RNN mode
        feature_norm = feature_current.reshape(labels.shape[0], -1)
        if normalization == 'min-max':
            feature_norm = scaler.fit_transform(feature_norm)
        if normalization == 'unit':
            feature_norm = normalize(feature_norm)
        feature_current = feature_norm.reshape(feature_current.shape)

        feature_list.append(feature_current)

        names = open(f'{path}/eq_week-{feature_type}-{course}/settings.txt', 'r').read()
        names = ast.literal_eval(names)['feature_names']
        names = [f"{'_'.join(name.split('_')[:-1])} {name.split('_')[-1].split(' ')[1]}" if 'function ' in name else name for name in names]
        names_list += names
        
    course_features = np.concatenate(feature_list, axis=2)
    # Train-val-test split
    x_train, x_test_v, y_train, y_test_v = train_test_split(course_features, labels, test_size=TEST_SIZE + VAL_SIZE, random_state=0, stratify=hard_fails['label-pass-fail'])
    x_test, x_val, y_test, y_val = train_test_split(x_test_v, y_test_v, test_size=VAL_SIZE/(TEST_SIZE + VAL_SIZE), random_state=0, stratify=y_test_v)

    
    return x_train, x_test, x_val, y_train, y_test, y_val, names_list



import pandas as pd
import numpy as np
from collections import Counter
import math


def dtree(train, criterion, max_depth=None, min_instances=2, target_impurity=0.0, class_col='class'):
    '''
    Input:
        train: training dataset, pandas dataframe
        criterion: Attribute selection used to find optimal split, 'gini' or 'entropy'
        max_depth: maximum depth of the tree
        min_instances: minimum number of heterogenous instances required to split
        target_impurity: target impurity for the tree

    Output:
        model: A decision tree model represented as a tuple of tuples. This tuple contains
            * feature / column name (splitting criterion)
            * feature value threshold (splitting criteria)
            * examples_in_split
            * majority class
            * impurity_score
            * depth
            * left_subtree (leading to examples where feature value <= test threshold)
            * right_subtree (leading to examples where feature value > test threshold

    This function builds a decision tree using the training dataset. The tree is built recursively
    '''
    # If
    if train is None or len(train) == 0:
        return None
    elif Counter(train[class_col]).most_common(1)[0][1] < min_instances or Counter(train[class_col]).most_common(1)[0][1] == len(train):
        best_col, best_v, best_meas = rachlins_best_split(train, class_col, criterion)
        majority = Counter(train[class_col]).most_common(1)[0][0]
        return (best_col, 
                best_v, 
                train, 
                majority, 
                best_meas, 
                None, 
                None)
    else:
        # Find the best split
        best_col, best_v, best_meas = rachlins_best_split(train, class_col, criterion)
        if type(best_v) == str or type(best_v) == bool:
            left_vals = train[train[best_col] == best_v]
            right_vals = train[train[best_col] != best_v]
            if len(left_vals) == len(train) or len(right_vals) == len(train):
                return (best_col, 
                        best_v, 
                        train, 
                        majority, 
                        best_meas, 
                        None, 
                        None)
        else:
            left_vals = train[train[best_col] <= best_v]
            right_vals = train[train[best_col] > best_v]
            if len(left_vals) == len(train) or len(right_vals) == len(train):
                majority = Counter(train[class_col]).most_common(1)[0][0]
                return (best_col, 
                        best_v, 
                        train, 
                        majority, 
                        best_meas, 
                        None, 
                        None)
        majority = Counter(train[class_col]).most_common(1)[0][0]
        majority_count = Counter(train[class_col]).most_common(1)[0][1]
        train_len = len(train)
        return (best_col, 
                best_v, 
                train, 
                majority, 
                best_meas, 
                dtree(left_vals, criterion, max_depth, min_instances, target_impurity, class_col=class_col), 
                dtree(right_vals, criterion, max_depth, min_instances, target_impurity, class_col=class_col))


def tree(L, min_vals=1):
    '''
    Buiild a binary tree but stop splitting if number of values falls below some threshold 
    '''
    if L is None or len(L) == 0:
        return None
    elif len(L) < min_vals:
        return (L[0], None, None, L)
    else:
        key = L[0]
        left_vals = [x for x in L[1:] if x <= key]
        right_vals = [x for x in L[1:] if x > key]
        return (key, tree(left_vals, min_vals), tree(right_vals, min_vals), L)
    
def total(cnt):
    return sum(cnt.values())

def gini(cnt):
    '''1 - sum(p^2)
    Takes a Counter object and returns the gini impurity
    '''
    tot = total(cnt)
    return 1- sum([(v/tot)**2 for v in cnt.values()])
    
def entropy(cnt):
    '''-sum(p*log(p))
    Takes a Counter object and returns the entropy
    '''
    tot = total(cnt)
    return -sum([(v/tot)*math.log2(v/tot) for v in cnt.values()])
    
def wavg(cnt1, cnt2, measure):
    tot1, tot2 = total(cnt1), total(cnt2)
    return (tot1*measure(cnt1) + tot2*measure(cnt2))/(tot1+tot2)

def rachlins_best_split(df, class_col, measure):
    best_col = 0
    best_v = ''
    best_meas = float("inf")
    
    for split_col in df.columns:
        if split_col != class_col:
            v, meas = rachlins_best_split_for_column(df, class_col, split_col, measure)
            if meas < best_meas:
                best_v = v
                best_meas = meas
                best_col = split_col
                
    return best_col, best_v, best_meas

def rachlins_best_split_for_column(data, class_col, split_col, measure):
    best_v = ''
    best_meas = float('inf')
    for v in set(data[split_col]):
        meas = evaluate_split(data, class_col, split_col, v, measure)
        if meas < best_meas:
            best_meas = meas
            best_v = v
    return best_v, best_meas

def evaluate_split(data, class_col, split_col, feature_val, measure):
    data1, data2 = data[data[split_col] == feature_val], data[data[split_col] != feature_val]
    cnt1, cnt2 = Counter(data1[class_col]), Counter(data2[class_col])
    return wavg(cnt1, cnt2, measure)

def depth(T):
    '''return max depth of a tree'''
    if T == None or len(T) == None:
        return -1
    else:
        return 1 + max(depth(left(T)), depth(right(T)))
    
def left(T):
    return T[1] if T is not None else None

def right(T):
    return T[2] if T is not None else None

import pandas as pd
import numpy as np
from collections import Counter
import math



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

import numpy as np
import sklearn.metrics

def accuracy(output, target):
    output = np.argmax(output, axis=1)
    return sklearn.metrics.accuracy_score(target, output)

def f1_score(output, target, average='macro'):
    output = np.argmax(output, axis=1)
    return sklearn.metrics.f1_score(target, output, average=average)

def mean_squared_error(output, target):
    return sklearn.metrics.mean_squared_error(target, output, multioutput='raw_values')

def ccc(output, target):
    mean_target = np.mean(target, axis=0)
    mean_output = np.mean(output, axis=0)
    var_target = np.var(target, axis=0)
    var_output = np.var(output, axis=0)
    cor = np.mean((output - mean_output) * (target - mean_target), axis=0)
    r = 2*cor / (var_target + var_output + (mean_target-mean_output)**2)
    return r[0], r[1]
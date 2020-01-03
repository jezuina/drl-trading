import numpy as np

def my_round(x):
    return np.round(x, 4)

def my_log(x):
    try:
        return np.log(x)
    except:
        return 0

def sum_abs(x):
    ''' Sum of absolute values '''
    return np.sum(np.abs(x))

def log_negative(x):
    if x > 0:
        return np.log(x)
    elif x == 0:
        return 0
    elif x < 0:
        return -np.log(-x)

def log_negative_on_array(x):
    l = []
    for i in range(len(x)):
        if x[i] > 0:
            l.append(np.log(x[i]))
        elif x[i] == 0:
            l.append(0)
        elif x[i] < 0:
            l.append(-np.log(-x[i]))
    
    return l
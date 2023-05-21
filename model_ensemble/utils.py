import sys


def list_argmax(array):
    min_array = -sys.maxsize
    idx = 0
    for i in range(len(array)):
        if min_array < array[i]:
            min_array = array[i]
            idx = i
    return idx, min_array


def softmax(score):
    exp = np.exp(score)
    sum_exp = np.sum(exp)
    exp_score = exp / sum_exp
    return exp_score

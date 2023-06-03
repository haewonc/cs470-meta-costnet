import numpy as np

def seq2instance(data, num_his, num_pred, offset=0):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred - offset + 1
    x = np.zeros((num_sample, num_his, dims))
    y = np.zeros((num_sample, num_pred, dims))
    for i in range(num_sample):
        x[i] = data[i: i + num_his, :]
        y[i] = data[i + offset + num_his: i + offset + num_his + num_pred, :]
    return x, y
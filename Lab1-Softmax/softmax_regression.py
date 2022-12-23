# coding=utf-8
import numpy as np


def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    count = 0
    for it in range(iters):
        loss_i = 0
        grad_i = np.zeros((theta.shape[0], theta.shape[1]))  # grad_i:[k*n]
        for i in range(x.shape[0]):
            xx = x[i].reshape(-1, 1)        # xx:[n*1]
            yy = y[:, i].reshape(-1, 1)     # yy:[k*1]
            ''' calculate theta_t * x[i]
                theta[k*n], 每一维是[1*n], xx[n*1] '''
            z = np.dot(theta, xx)
            ''' calculate softmax[i] '''
            row_max = np.max(z)
            z = z - row_max
            y_pre_i = np.exp(z) / np.sum(np.exp(z))   # y_pre[k*1]
            ''' calculate loss[i] '''
            loss_i += np.sum(yy * np.log(y_pre_i))
            ''' calculate grad[i] '''
            grad_i += np.dot((y_pre_i-yy), xx.T)
        ''' calculate loss '''
        f = -(1/y.shape[1]) * loss_i
        print("loss: ", f)
        count += 1
        print("count: ", count)
        ''' calculate grad '''
        g = -(1/y.shape[1]) * grad_i
        ''' update theta '''
        theta += alpha * g
    return theta
    

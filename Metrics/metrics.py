from __future__ import division

import numpy as np

def l2_loss(yhat,y):
    return np.mean(sum((yhat-y)**2))

def l2_loss_gradient(yhat,y):
    return y - yhat

def gini_impurity():
    pass

def SE(yhat,y):
    return sum((y-yhat)**2)

def RMSE(yhat,y):
    return np.sqrt(SE(yhat,y)/len(y))

def RMSPE(yhat,y):
    yhat = yhat[y != 0]
    y = y[y != 0]
    return np.sqrt(np.mean((y-yhat)**2/(y**2)))

def SPE_gradient(yhat,y):
    answer = np.zeros(len(y))
    answer[y != 0] = (y[y != 0] - yhat[y != 0])/(y[y != 0]**2)
    return answer

def l2_gamma(y, prediction, update):
    return (y - prediction).dot(update)/(sum(update**2))

def mspe_gamma(y, prediction, update):
    prediction = prediction[y != 0]
    update = update[y != 0]
    y = y[y != 0]
    return update.dot((y-prediction)/(y**2))/update.dot(update/(y**2))
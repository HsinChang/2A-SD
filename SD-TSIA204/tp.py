import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import inv
from matplotlib import rc

file_name = "data_dm3.csv"
df = pd.read_csv(file_name, delimiter=",", sep="\n", header=None)
Y = df.iloc[:, -1]
X = df.iloc[:, :-1]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

from scipy.optimize import leastsq

def model(p,x):
    a, b, c = p
    u = x[:,0]
    v = x[:,1]
    return (a*u**2 + b*v + c)

def resid_func(p, y, x):
    a,b,c = p
    err = y - model(p,x)
    return err

p0 = np.array([2,3,4]) #initial guess of a, b, c

p = leastsq(resid_func, p0, args = (X_sel, Y_train))[0]

print(p)

'''
from sklearn.utils import shuffle

p = X.shape[1]

from sklearn.linear_model import LassoCV
reg = LassoCV().fit(X, Y)
print(np.nonzero(reg.coef_)[0])
'''

'''
X_temp, Y_temp = shuffle(X_train, Y_train)
X_list = []
Y_list = []
fold_sizes = np.full(4, X_train.shape[0] // 4, dtype=np.int)
fold_sizes[:X_train.shape[0] % 4] += 1
print("Number of observations in each fold: " + str(fold_sizes))
current = 0
for i in range(0, 4):
    start, stop = current, current + fold_sizes[i]
    X_list.append(X_temp.iloc[start: stop])
    Y_list.append(Y_temp.iloc[start: stop])
    current = stop

from numpy.linalg import norm

def risk_ridge(lmb):
    risk = 0
    for i in range (0, 4):
        a = [0, 1, 2, 3]
        a.remove(i)
        temp_X_train = np.asmatrix(pd.concat([X_list[a[0]], X_list[a[1]], X_list[a[2]]]))
        temp_Y_train = np.asmatrix(pd.concat([Y_list[a[0]], Y_list[a[1]], Y_list[a[2]]]))
        temp_X_test = np.asmatrix(np.ascontiguousarray(X_list[i]))
        temp_Y_test = np.asmatrix(np.ascontiguousarray(Y_list[i]))
        I_p = np.matrix(np.eye(p))
        theta_rdg = inv(temp_X_train.T * temp_X_train + temp_X_train.shape[0]*lmb*I_p)*(temp_X_train.T)*(temp_Y_train.T)
        temp_Y_pre = temp_X_test * theta_rdg
        risk += norm(temp_Y_test.T - temp_Y_pre) ** 2
    return risk/4
'''

'''
                for k in range (0, 4):
            if (k != i):
                temp_X_train.extend(X_list[k])
                temp_Y_train.extend(Y_list[k])
            else:
                temp_X_test.extend(X_list[k])
                temp_Y_test.extend(Y_list[k])
        temp_X_train = np.concatenate(temp_X_train, axis=0)
        temp_X_train = np.reshape(temp_X_train, (X_temp.shape[0]-fold_sizes[i], X_temp.shape[1]))
        temp_X_test = np.concatenate(temp_X_test, axis=0)
        temp_X_test = np.reshape(temp_X_test, (fold_sizes[i], X_temp.shape[1]))
        '''

print("")

'''
from scipy.stats import norm

X_aug = np.column_stack((np.ones((X_train.shape[0], 1)), X_train))

p = X_aug.shape[1]
n = X_aug.shape[0]

test = np.zeros((p, p))
residuals = Y_train
p_val_mem = np.zeros(p)
p_val = np.zeros((p, p))

var_sel = []
var_remain = list(range(p))
in_test = []

reg = LinearRegression()

for k in range(p):
    resids_mem = np.zeros((p, n))

    for i in var_remain:
        xtmp = X_aug[:, [i]]
        reg.fit(xtmp, residuals)
        xx = np.sum(X_aug[:, [i]] ** 2)
        resids_mem[i, :] = reg.predict(xtmp) - residuals
        sigma2_tmp = np.sum(resids_mem[i, :] ** 2) / xx
        test[k, i] = np.sqrt(n) * np.abs(reg.coef_) / (np.sqrt(sigma2_tmp))
        p_val[k, i] = 2 * (1 - norm.cdf(test[k, i]))

    best_var = np.argmax(test[k, :])
    var_sel.append(best_var)
    residuals = resids_mem[best_var, :]
    p_val_mem[k] = p_val[k, best_var]
    var_remain = np.setdiff1d(var_remain, var_sel)
idx_sel = np.array(var_sel)[p_val_mem < .1]
X_sel = X_train.iloc[:, idx_sel]
reg_fvs = LinearRegression(fit_intercept = True).fit(X_sel, Y_train)
r_fvs = sum((reg_fvs.predict(X_test.iloc[:, idx_sel]) - Y_test) ** 2)
'''

print("")
'''
var_X_train = np.cov(X_train.T)
u, s, vh = np.linalg.svd(var_X_train)
from sklearn.linear_model import LinearRegression
reg_ols = LinearRegression().fit(X_train, Y_train)
reg_pca = LinearRegression().fit(np.asmatrix(X_train) * np.asmatrix(u[:,0:60]), Y_train)
from sklearn import preprocessing
X_pca = np.asmatrix(X_train) * np.asmatrix(u[:,0:60])
X_pca_scaled = preprocessing.scale(X_pca)
Y_ols = reg_ols.predict(X_train)
Y_pca = reg_pca.predict(X_pca)
aaa = Y_ols - Y_train
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.hist(np.hstack(Y_ols - Y_train), )
ax2 = fig.add_subplot(212)
ax1.hist(np.hstack(Y_pca - Y_train))
'''

'''
import random

def get_split_index(a):
    fold_sizes = np.full(a, X_train.shape[0] // a, dtype=np.int)
    fold_sizes[:X_train.shape[0] % a] += 1
    index = np.arange(X_train.shape[0]-1)
    random.shuffle(index)
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        yield index[start:stop].tolist()
        current = stop

i_fold1, i_fold2, i_fold3, i_fold4 = get_split_index(4)
X_fold1, X_fold2, X_fold3, X_fold4 = X_train.iloc[i_fold1], X_train.iloc[i_fold2], X_train.iloc[i_fold3], X_train.iloc[i_fold4]
Y_fold1, Y_fold2, Y_fold3, Y_fold4 = Y_train.iloc[i_fold1], Y_train.iloc[i_fold2], Y_train.iloc[i_fold3], Y_train.iloc[i_fold4]
'''


'''
from sklearn.linear_model import LassoCV
reg = LassoCV().fit(X, Y)
print(np.nonzero(reg.coef_)[0])
'''

print("")

import numpy as np
from scipy.optimize import leastsq
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

X_train = np.array(X_train)
Y_train = np.array(Y_train).reshape((-1,1))
data =np.array(
[[-0.042780748663101636, -0.0040771571786609945, -0.00506567946276074],
[0.042780748663101636, -0.0044771571786609945, -0.10506567946276074],
[0.542780748663101636, -0.005771571786609945, 0.30506567946276074],
[-0.342780748663101636, -0.0304077157178660995, 0.90506567946276074]])

coefficient = data[:,0:2]
dependent = data[:,-1]

def model(p,x):
    a,b,c = p
    u = x[:,0]
    v = x[:,1]
    return (a*u**2 + b*v + c)

def residuals(p, y, x):
    a,b,c = p
    err = y - model(p,x)
    return err

p0 = np.array([2,3,4])#some initial guess



p = leastsq(residuals, p0, args=(dependent, coefficient))[0]

print(p)
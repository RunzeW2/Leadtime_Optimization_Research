import numpy as np
import numpy.linalg as la
from scipy.stats import erlang
#holding cost h
h = [10,20]
#penalty cost p
p = 50
#number of stages N
N = 2
def serial_sgd(N, h, p, max_iter):
    T_initial = np.zeros((N))
    T = T_initial.copy()
    tol = 0
    for i in range(N):
        T[i] = np.random.normal(10,1)
    for i in range(max_iter):
        T_gradient = np.random.normal(10,2,size=2)
        tau = np.random.normal(10,1,size=2)
        for j in range(N):
            diff = 0
            diff_list = np.zeros((N, 1))
            for k in range(N):
                diff += (T[k] - tau[k])
                diff_list[k] = diff
        for j in range(N):
            if diff_list[j] > 0:
                T_gradient += h[j]
            else:
                T_gradient -= p
        stepsize = 1e-4
        T_old = T.copy()
        T -= stepsize * T_gradient.T
        tol = T - T_old
    return T,tol

#Final_T = serial_sgd(N, h, p, max_iter=10000)

print(serial_sgd(N, h, p, max_iter=100000))
#def ExpectedValue(funct,max_iter=100):
#    value = 0
#    for i in range(max_iter):
#        value += funct
#    value = value / max_iter
#    return value
#T_test = ExpectedValue(Final_T)
#T_tol = ExpectedValue(Tol_T)
#def cost(T_test):
#    cost = 20 * T_test[0] + h[0] * (max((T_test[0] - 10), 0)) + 50 * (max((10 - T_test[0]), 0))+20 * (T_test[1]-(T_test[0] - 10)) + h[1] * (max((T_test[1] - 10), 0)) + 50 * (max((10 - T_test[1]), 0))
#    return cost
#c_tol = cost(T_test)-cost(T_tol)


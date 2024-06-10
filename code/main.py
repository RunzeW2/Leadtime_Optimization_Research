import numpy as np
import algorithms as alg

# Initialization
N = [2, 3, 3]
M = len(N)

h = []
shape = []
scale = []
for i in range(M):
    h_list = []
    shape_list = []
    scale_list = []
    for j in range(N[i]):
        h_list.append(10)
        shape_list.append(1)
        scale_list.append(1)
    h.append(h_list)
    shape.append(shape_list)
    scale.append(scale_list)

p = 500

H = [[70, 60], [30, 20, 10], [30, 20, 10]]

T_ASML = alg.ASML(N, shape, scale)
T_sgd = alg.heuristic(N, H, p, shape, scale)

W_ASML, holding_cost_ASML, back_ordering_cost_ASML, ontime_count_ASML = alg.simulation(T_ASML, N, H, p, shape, scale)
W_sgd, holding_cost_sgd, back_ordering_cost_sgd, ontime_count_sgd = alg.simulation(T_sgd, N, H, p, shape, scale)

cost_ASML = np.mean(holding_cost_ASML) + np.mean(back_ordering_cost_ASML)
cost_sgd = np.mean(holding_cost_sgd) + np.mean(back_ordering_cost_sgd)

print(cost_ASML, cost_sgd)

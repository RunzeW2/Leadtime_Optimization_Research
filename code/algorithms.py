import numpy as np
from scipy.stats import erlang
from scipy.stats import norm


def ASML(N, shape, scale):
    """
    N: M x 1 vector, length of (M-1) subassemblies and final assembly process (index 0)
    shape: shape of Erlang distribution (throughput time)
    sacle: sacle of Erlang distribution
    T: planned leadtimes of each stage
    """
    M = len(N)
    T = []
    for i in range(M):
        T_list = []
        for j in range(N[i]):
            sample = erlang.rvs(shape[i][j], loc=0, scale=scale[i][j], size=100)
            mean = np.mean(sample)
            std = np.std(sample)
            T_list.append(norm.ppf(0.8, loc=mean, scale=std))
        T.append(T_list)

    return T


def serial_sgd(N, h, p, shape, scale, W, index_W, max_iter=1000):
    """
    N: length of the serial systems
    h: N x 1 vector, local holding cost of each process
    p: penalty cost
    shape: shape of Erlang distribution (throughput time)
    sacle: sacle of Erlang distribution
    max_iter: maximum number of iterations
    W: waiting time
    index: stage
    """

    # Initialization
    T_initial = np.zeros((N, 1))
    T = T_initial.copy()

    for i in range(max_iter):
        T_gradient = np.zeros((N, 1))

        # generate sample of throughput time
        tau = erlang.rvs(shape, loc=0, scale=scale, size=N)
        tau[index_W] += W

        index = np.zeros((N, N))
        G = np.zeros((N, 1))

        for j in range(N):
            diff = 0
            diff_list = np.zeros((N, 1))
            for k in range(j, N):
                diff += (T[k] - tau[k])
                diff_list[k] = diff
            min_index = np.argmin(diff_list[j: N]) + j
            index[j:min_index + 1, j] = 1
            G[j] = np.min(diff_list[j: N])

        if G[0] > 0:
            T_gradient += h[0] * index[:, 0].reshape((N, 1))
        else:
            T_gradient -= p * index[:, 0].reshape((N, 1))

        for j in range(1, N):
            if G[j] > 0:
                T_gradient += h[j] * index[:, j].reshape((N, 1))

        stepsize = 1e-4
        T -= stepsize * T_gradient

    return T


def heuristic(N, H, p, shape, scale, max_iter=100):
    """
    N: M x 1 vector, length of (M-1) subassemblies and final assembly process (index 0)
    H: local holding cost of each process
    p: penalty cost
    shape: shape of Erlang distribution (throughput time)
    sacle: sacle of Erlang distribution
    T: planned leadtimes of each stage
    """

    # initialization
    M = len(N)
    W = np.zeros((M - 1, 1))

    for i in range(max_iter):
        T0 = np.zeros((N[0], 1))
        T = []

        for j in range(1, M):
            H_serial = H[0] + H[j]
            shape_serial = shape[0] + shape[j]
            scale_serial = scale[0] + scale[j]

            # solving the new serial system
            T_serial = serial_sgd(N[0] + N[j], H_serial, p, shape_serial, scale_serial, W[j - 1], N[0])

            T.append(T_serial[N[0]:(N[0] + N[j])])
            T0 += T_serial[0: N[0]]

        T.insert(0, T0 / (M - 1))

        W_data = simulation(T, N, H, p, shape, scale, iterations=100)[0]

        W_before = W.copy()
        W = np.mean(W_data, axis=1)

        print(W)

        rel_error = np.linalg.norm(W - W_before, 2) / np.linalg.norm(W, 2)
        print(rel_error)
        if rel_error < 1e-4:
            break

    return T


def simulation(T, N, H, p, shape, scale, iterations=1000):
    """
    N: M x 1 vector, length of (M-1) subassemblies and final assembly process (index 0)
    H: local holding cost of each process
    p: penalty cost
    shape: shape of Erlang distribution (throughput time)
    sacle: sacle of Erlang distribution
    T: planned leadtimes of each stage
    """

    M = len(N)
    W_data = np.zeros((M - 1, iterations))
    holding_cost = np.zeros(iterations)
    back_ordering_cost = np.zeros(iterations)
    ontime_count = 0
    for i in range(iterations):
        L = np.zeros((M - 1, 1))
        for j in range(1, M):
            for k in range(N[j] - 1, -1, -1):
                tau = erlang.rvs(shape[j][k], loc=0, scale=scale[j][k])
                holding_cost[i] += H[j][k] * max(T[j][k] - L[j - 1], tau)
                L[j - 1] = max(0, L[j - 1] + tau - T[j][k])
        tardiness = max(L)
        for j in range(M - 1):
            W_data[j, i] = tardiness - L[j]
        for j in range(N[0] - 1, -1, -1):
            tau = erlang.rvs(shape[0][j], loc=0, scale=scale[0][j])
            holding_cost[i] += H[0][j] * max(T[0][j] - tardiness, tau)
            tardiness = max(0, tardiness + tau - T[0][j])
        if tardiness < 1e-8:
            ontime_count += 1
        back_ordering_cost[i] = tardiness * p

    return W_data, holding_cost, back_ordering_cost, ontime_count

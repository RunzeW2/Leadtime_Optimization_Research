import numpy as np
#T is observation input:vector for leadtimes T
#C is observation output:vector for cost C

def sgd(
    gradient, T, C, start, learn_rate=0.1, batch_size=1, n_iter=50,
    tolerance=1e-06, dtype="float64", random_state=None
):
    #Convert Variables
    dtype_ = np.dtype(dtype)
    vector = np.array(start, dtype=dtype_)
    learn_rate = np.array(learn_rate, dtype=dtype_)
    batch_size = int(batch_size)
    n_iter = int(n_iter)
    tolerance = np.array(tolerance, dtype=dtype_)

    # Converting x and y to NumPy arrays
    T, C = np.array(T, dtype=dtype_), np.array(C, dtype=dtype_)
    n_obs = T.shape[0] #returns number of samples in x.
    TC = np.c_[T.reshape(n_obs, -1), C.reshape(n_obs, 1)]

    #randomization
    seed = None if random_state is None else int(random_state)
    rng = np.random.default_rng(seed=seed)

    # Performing the gradient descent loop
    for _ in range(n_iter):
        # Shuffle x and y, use a stochastic input/output
        rng.shuffle(TC)

        # Performing minibatch moves
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

            # Recalculating the difference
            grad = np.array(gradient(x_batch, y_batch, vector), dtype_)
            diff = -learn_rate * grad

            # Checking if the absolute difference is small enough
            if np.all(np.abs(diff) <= tolerance):
                break

            # Updating the values of the variables
            vector += diff

    return vector if vector.shape else vector.item()

#Generate Randomized input variable T(100 input data points).
N = 3
M = 3
h = 10 * np.ones((M+1, N))
H = [[100, 110, 120], [10, 20, 30], [10, 20, 30], [10, 20, 30]]
p = 100
shape = 1
scale = 1
from scipy.stats import erlang
from scipy.stats import norm
T = np.zeros((100, M+1, N))
for k in range (100):
    for i in range(M+1):
        for j in range(N):
            sample = erlang.rvs(shape, size=100)
            mean = np.mean(sample)
            std = np.std(sample)
            T[k][i][j] = norm.ppf(0.8, loc = mean, scale = std)
print(T)

#Calculate Cost Using T.


#Run sgd function for minimal value of T







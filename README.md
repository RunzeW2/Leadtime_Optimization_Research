# README

## Overview

This repository contains the implementation and experiments for optimizing assembly systems using Stochastic Gradient Descent (SGD) and Dynamic Programming approaches. The focus is on minimizing the cost function for a multi-stage assembly system, considering both holding and penalty costs.

## Introduction

### Stochastic Gradient Descent (SGD)
Stochastic Gradient Descent (SGD) is an iterative optimization algorithm used to minimize an objective function. It is particularly useful for large-scale and high-dimensional optimization problems. In our context, SGD is applied to optimize the planned lead times in an assembly system to minimize the overall cost.

### Dynamic Programming
Dynamic Programming is a method used for solving complex problems by breaking them down into simpler subproblems. It is used to solve the cost function optimization problem by considering the dependencies between different stages of the assembly process.

## Cost Function

The main cost function we aim to optimize is given by:

$$
C(T) = E \left[ \sum_{m=1}^{M} \sum_{j=1}^{N_m} h_{mj} \left( \sum_{i=1}^{N_0} T_{0i} + \sum_{i=1}^{N_m} T_{mi} - \sum_{i=j+1}^{N_m} (\tau_{mi} + E_{mi}) \right) + \sum_{j=1}^{N_0} h_{0j} \left( \sum_{i=1}^{N_0} T_{0i} - W_0 - \sum_{i=j+1}^{N_0} (\tau_{0i} + E_{0i}) \right) + (H_{01} + p)L_{01} \right]
$$

### Simplified One-Stage Assembly System

For a one-stage assembly system, the cost function simplifies to:

$$
C(T) = E \left[ \sum_{m=0}^{M} H_m T_m - h_0 W_0 + (H_0 + p)L_0 \right]
$$

## Optimization Approach

### Stochastic Gradient Descent (SGD)

Instead of solving the generalized newsvendor equations, we apply SGD to directly optimize the simplified cost function. The gradients for the cost function are computed and used to iteratively update the planned lead times.

**SGD Algorithm:**

1. Initialize the planned lead times $$\(T_1\)$$ arbitrarily and set the step size $$\(\eta_k\)$$.
2. At each iteration $$\(k\)$$, sample the throughput times $$\(\{\tau^k_m\}\)$$ and update the lead times:
$$\[ T_{k+1} = T_k - \eta_k s_k(T_k, \tau^k) \]$$
3. Stop the algorithm when the difference between successive iterations is small enough.

### Dynamic Programming

Dynamic Programming is used to solve the cost function optimization problem by considering the dependencies between different stages of the assembly process. This approach is applied to a serial system with multiple stages.

## Experiments and Results
$$
We conducted experiments on an assembly system with two \(N_m\)-stage processes delivering subassemblies and a \(N_0\)-stage assembly process. The parameters were set as follows:
- \(M = 2\)
- \(N_0 = 2\)
- \(N_m = 3\)
- Marginal cost \(h_{mj} = 10\)
- Penalty cost \(p = 500\)
- Lead time follows the Erlang distribution: \(\tau_{mj} \sim \text{Erlang}(1, 1)\)
$$
### Performance Evaluation

We ran 10,000 repeated simulations and averaged the cost functions for different optimization methods. The results are summarized below:

| Method | On-time Delivery % | Holding Cost | Penalty Cost | Percentage % |
|--------|--------------------|--------------|--------------|--------------|
| ASML   | 75.6               | 486.425      | 162.351      | 113.976      |
| NV     | 87.0               | 469.314      | 99.907       | 100.000      |
| HSGD   | 86.8               | 469.827      | 100.013      | 100.109      |
| SGD    | 88.2               | 465.764      | 97.336       | 98.925       |

## Conclusion

The experimental results demonstrate that the SGD approach is effective in optimizing the planned lead times, achieving the lowest penalty cost and a high on-time delivery percentage. Dynamic Programming also provides a robust framework for solving the cost function optimization problem in serial systems.

## References

Atan, Z., de Kok, T., Dellaert, N. P., van Boxel, R., & Janssen, F. (2016). Setting planned leadtimes in customer-order-driven assembly systems. *Manufacturing & Service Operations Management*, 18(1), 122–140.

## Author
Minda Zhao, Runze Wang



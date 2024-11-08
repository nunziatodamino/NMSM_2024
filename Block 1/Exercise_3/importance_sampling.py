import numpy as np
import scipy.stats as stats
import numba

def crude_monte_carlo (function, left_bound, right_bound, iterations):
    tmp = function(np.random.uniform(left_bound, right_bound, size = iterations))
    return (right_bound - left_bound ) * np.sum(tmp) / iterations    


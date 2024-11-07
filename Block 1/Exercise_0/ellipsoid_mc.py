import numpy as np
import random

def analytic_ellipsoid_octant (a:np.float64, b:np.float64, c:np.float64) -> np.float64:
    return 1/8 * 4/3 * np.pi * a * b * c 

def integration_box_volume (a:np.float64, b:np.float64, c:np.float64) -> np.float64:
    return a * b * c

def monte_carlo_ellipsoid_octant (a:np.float64, b:np.float64, c:np.float64, iteration_number:np.int32) -> (list, list) :
    iterations = []
    error = []
    counter = 0
    for i in range(iteration_number):
        x = a * random.random()
        y = b * random.random()
        z = c * random.random()
        ellipsoid = (x**2 / a**2) + (y**2 / b**2) + (z**2 / c**2)
        if ellipsoid < 1 :
            counter += 1
        if i % 100 == 0 : 
            iterations.append(i)
            volume_estimate = counter / iteration_number * integration_box_volume(a, b, c)
            error.append(abs(analytic_ellipsoid_octant(a, b, c) - volume_estimate))
    return iterations, error
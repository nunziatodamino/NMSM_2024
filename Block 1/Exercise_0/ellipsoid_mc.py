import numpy as np
import numba

@numba.njit
def analytic_ellipsoid_octant (a:np.float64, b:np.float64, c:np.float64) -> np.float64:
    return 1/8 * 4/3 * np.pi * a * b * c 

@numba.njit
def integration_box_volume (a:np.float64, b:np.float64, c:np.float64) -> np.float64:
    return a * b * c

@numba.njit
def monte_carlo_ellipsoid_octant (a:np.float64, b:np.float64, c:np.float64, iteration_number:np.int32) -> np.float64 :
    counter = 0
    for _ in range(iteration_number):
        x = a * np.random.random()
        y = b * np.random.random()
        z = c * np.random.random()
        ellipsoid = (x**2 / a**2) + (y**2 / b**2) + (z**2 / c**2)
        if ellipsoid < 1 :
            counter += 1
    return counter / iteration_number * integration_box_volume(a, b, c)
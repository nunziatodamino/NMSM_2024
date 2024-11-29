import numpy as np

def lennard_jones_potential(epsilon : float, sigma : float, sigma_cut : float, distance : float) -> float: 
    if distance >= sigma_cut:
        return 0
    return 4 * epsilon * ( (sigma / distance)**12 - (sigma / distance)**6 )

def energy_tail_correction(rho : float, sigma : float, sigma_cut : float) -> float:
    return (8 / 3) * np.pi * rho * ( (1 / 3) * (sigma / sigma_cut)**9 -(sigma / sigma_cut)**3 )

def pressure_tail_correction(rho : float, sigma : float, sigma_cut : float) -> float:
    return (16 / 3) * np.pi * rho**2 * ( (2 / 3) * (sigma / sigma_cut)**9 -(sigma / sigma_cut)**3 )    

def position_PBC(position : float, box_length : float) -> float:
    return position - np.floor(position / box_length) * box_length



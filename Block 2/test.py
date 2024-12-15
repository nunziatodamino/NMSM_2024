import numpy as np

a = np.array([1,2,3])
b = np.array([0.1,0.2, 0.3, 0.4, 0.5, 0.6])
v = np.array([1,2,3,4,5,6])

c = np.outer(b, a)
 
print(v * c)
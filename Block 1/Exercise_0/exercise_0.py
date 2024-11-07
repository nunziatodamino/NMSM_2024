import matplotlib.pyplot as plt
import ellipsoid_mc as mc

# program parameters
X_SEMI_AXIS = 3.0
Y_SEMI_AXIS = 2.0
Z_SEMI_AXIS = 2.0
ITERATIONS = 10000

iterations, error = mc.monte_carlo_ellipsoid_octant(X_SEMI_AXIS, Y_SEMI_AXIS, Z_SEMI_AXIS, ITERATIONS)

plt.plot(iterations, error, marker='o', linestyle='-', color='b')

plt.title('Error of the Monte Carlo simulation as a function of the number of iterations')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()

# New ellipsoid parameters
a1 = 3.0
b1 = 1.0
c1 = 1.0

iterations1, error1 = mc.monte_carlo_ellipsoid_octant(a1, b1, c1, ITERATIONS)

plt.plot(iterations, error, marker='o', linestyle='-', color='b')
plt.plot(iterations1, error1, marker='o', linestyle='-', color='r')

plt.title('Error of the Monte Carlo simulation as a function of the number of iterations')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()
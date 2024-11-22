import numpy as np

# part 1

P = np.array([
    [1/2, 1/3, 1/6],
    [3/4, 0, 1/4],
    [0, 1, 0]
])

print(np.dot(P,P))

max_steps=1000
A=np.eye(3)

for n in range(max_steps):
    A=np.dot(A,P)

print(A)

# part 2

P = np.array([
    [0, 1, 0],
    [1/6, 1/2, 1/3],
    [0, 2/3, 1/3]
])

Q = np.dot(P,P)

print(np.dot(Q,P))

max_steps=1000
A=np.eye(3)

for n in range(max_steps):
    A=np.dot(A,P)

print(A)
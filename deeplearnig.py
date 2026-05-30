import numpy as np
import matplotlib.pyplot as pypy
def machine (x1, x2):
    nyu = np.array ([x1, x2])
    w1  = np.array ([[1, 2, 3], [4, 5, 6]])
    b1  = 5
    a1  = np.dot (nyu, w1) - b1
    z1  = np.maximum(0, a1)
    w2  = np.array ([[7, 8], [9, 10],[11, 12]])
    b2  = 10
    a2  = np.dot (z1, w2) - b2
    z2  = np.maximum(0, a2)
    w3  = np.array ([[13, 14], [15, 16]])
    b3  = 20
    a3  = np.dot (z2, w3) - b3
    z2  = np.maximum(0, a3)
    return (z2)
x1 = float(input("x1 = ?"))
x2 = float(input("x2 = ?"))
y  = machine(x1, x2)
print(y)
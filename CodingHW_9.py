import numpy as np
import scipy.integrate

a = 0.7
b = 1
tau = 12
I = lambda t: (1/10) * (5 + np.sin((np.pi * t)/10))
f = lambda t, v: np.array([v[0] - (1/3) * (v[0] ** 3) - v[1] + I(t), (a + v[0] - b*v[1])/tau])
v0 = np.array([1, 0])

T = 100
dt = 0.5
t = np.arange(0, T + dt, dt)
n = t.size

V = np.zeros((2, n))
V[:, 0] = v0
for k in range(n-1):
    f1 = f(t[k], V[:, k])
    V[:, k + 1] = V[:, k] + dt * f(t[k] + dt / 2, V[:, k] + (dt / 2) * f1)
x = V[0, :]
y = V[1, :]
A1 = x.reshape(1, 201)
# print("A1 = ", A1)

index = np.argmax(V[0, :20])
A2 = t[index]
print("A2 = ", A2)

index = np.argmax(V[0, 80:100])
A3 = t[index + 80]
print("A3 = ", A3)

A4 = 1 / (A3 - A2)
print("A4 = ", A4)

dt = 0.5
T = 100
t = np.arange(0, T + dt, dt)
n = t.size

V = np.zeros((2, n))
V[:, 0] = v0
for k in range(n - 1):
    f1 = f(t[k], V[:, k])
    f2 = f(t[k] + dt / 2, V[:, k] + (dt / 2) * f1)
    f3 = f(t[k] + dt / 2, V[:, k] + (dt / 2) * f2)
    f4 = f(t[k] + dt, V[:, k] + dt * f3)
    V[:, k + 1] = V[:, k] + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
x = V[0, :]
y = V[1, :]
A5 = x.reshape(1, 201)
# print("A5 = ", A5)

index = np.argmax(V[0, :20])
A6 = t[index]
# print("A6 = ", A6)

index = np.argmax(V[0, 80:100])
A7 = t[index + 80]
# print("A7 = ", A7)

A8 = 1 / (A7 - A6)
print("A8 = ", A8)

########## Problem 2 #########
x0 = 1
xT = 0.5
T = 6
dt = 0.1
t = np.arange(0, T + dt, dt)
n = t.size

v = -2 * np.ones(n - 2)
u = np.ones(n - 3)
A = (1 / dt ** 2) * (np.diag(v) + np.diag(u, 1) + np.diag(u, -1))
I = np.eye(n - 2)
A9 = A.copy() + I.copy()
# print("A9 = ", A9)
b = 5 * np.cos(4 * t[1: -1])
b[0] = b[0] - x0/ (dt ** 2)
b[-1] = b[-1] - xT/ (dt ** 2)
b = b.reshape((-1, 1))
A10 = b.copy()
# print("A10 = ", A10)

x_int = scipy.linalg.solve(A + I, b)
x = np.zeros(n)
x[0] = x0
x[1:-1] = x_int.reshape(-1)
x[-1] = xT
A11 = x.reshape(61, 1)
# print("A11 = ", A11)

C1 = ((1/2) + (1/3) * np.cos(24) - (4/3) * np.cos(6)) / np.sin(6)
C2 = 4/3
x_true = lambda t: C1 * np.sin(t) + C2 * np.cos(t) - (1/3) * np.cos(4*t)
A12 = np.max(np.abs(x - x_true(t)))
print("A12 = ", A12)

x0 = 1
xT = 0.5
T = 6
dt = 0.01
t = np.arange(0, T + dt, dt)
n = t.size

v = -2 * np.ones(n - 2)
u = np.ones(n - 3)
A = (1 / dt ** 2) * (np.diag(v) + np.diag(u, 1) + np.diag(u, -1))
I = np.eye(n - 2)
b = np.zeros(n - 2)
b = 5 * np.cos(4 * t[1: -1])
b[0] = b[0] - x0/ (dt ** 2)
b[-1] = b[-1] - xT/ (dt ** 2)

x_int = scipy.linalg.solve(A + I, b)
x = np.zeros(n)
x[0] = x0
x[1:-1] = x_int.reshape(-1)
x[-1] = xT
A13 = x.reshape(601, 1)
# print("A13 = ", A13)

A14 = np.max(np.abs(x - x_true(t)))
print("A14 = ", A14)

A15 = A12/A14
print("A15 = ", A15)


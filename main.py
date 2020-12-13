from fw import FW
from numpy import array
from numpy.linalg import norm


def min_func(x):
    return norm(x)**2/2


def min_grad_func(x):
    return x


points = [array([-1, 0]), array([1, 0]), array([0, 1])]
minimizer = array([0., 0.])
solver = FW(min_func, min_grad_func, points, minimizer=minimizer)
minimizer = solver.boostfw(y=array([-10, -1]), delta=1e-3, T=100)

print(minimizer['x'])

print(minimizer['error'])
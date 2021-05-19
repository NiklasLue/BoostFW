from fw import FW
from numpy import array
from numpy.linalg import norm


def min_func(x):
    return norm(x)**2/2


def min_grad_func(x):
    return x


points = [array([-1., 0.]), array([1., 0.]), array([0., 1.])]
minimizer = array([0., 0.])
solver = FW(min_func, min_grad_func, points, minimizer=minimizer)

minimizer_fw = solver.fw(x=array([1., 0.]), delta=1e-3)
print("---- FW ----")
print("Minimizer found:\t", minimizer_fw['x'])
print("Error:\t\t\t\t", minimizer_fw['error'][-1])
print("Iterations:\t\t\t", minimizer_fw['iterations'])

minimizer_boost = solver.boostfw(y=array([1, 0]), delta=1e-3, T=100)
print("---- BoostFW ----")
print("Minimizer found:\t", minimizer_boost['x'])
print("Error:\t\t\t\t", minimizer_boost['error'][-1])
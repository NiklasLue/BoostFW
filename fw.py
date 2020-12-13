import warnings

from numpy import dot, zeros, argmax, argmin, all, array, seterr, isclose
from numpy.linalg import norm

seterr(all='warn')
warnings.filterwarnings('error')


class FW:
    def __init__(self, func, grad_func, points, minimizer=None):
        self.func = func
        self.grad_func = grad_func
        self.points = points
        self.minimizer = minimizer

    def align(self, x, y):
        if all(isclose(y, zeros(y.shape[0]), atol=1e-15)):
            return -1
        else:
            return dot(x, y)/(norm(x)*norm(y))

    def step_func(self, t):
        return 2/(t+2)

    def boostfw(self, y, delta, T=10, K=10):
        if self.minimizer is not None:
            error = []
        grad_y = self.grad_func(y)
        # x = self.points[argmin([dot(grad_y, v) for v in self.points])]
        x = array([0.5, 0.5])
        for t in range(T):
            d = zeros(y.shape[0])
            mu = 0  # had to rename since lambda is a system function
            flag = False

            neg_grad_x = -self.grad_func(x)

            for k in range(K):
                d_norm = norm(d)
                r = neg_grad_x - d
                v = self.points[argmax([dot(r, v) for v in self.points])]
                if d_norm != 0.0:
                    u = [v-x, -d/d_norm]
                    u_argmax = argmax([dot(r, temp) for temp in u])
                    u = u[u_argmax]
                else:
                    u_argmax = 0
                    u = v - x
                if all(u == 0.):
                    coefficient = 1
                else:
                    coefficient = dot(r, u) / norm(u) ** 2

                d_prime = d + coefficient*u

                if self.align(neg_grad_x, d_prime) - self.align(neg_grad_x, d) >= delta:
                    d = d_prime
                    mu = mu + coefficient if u_argmax == 0 else mu*(1-coefficient/d_norm)
                else:
                    flag = True
                    break

            K = k if flag else K
            try:
                g = d/mu
            except RuntimeWarning:
                if mu == 0:
                    break
                else:
                    raise
            x = x + self.step_func(t)*g
            if self.minimizer is not None:
                error.append(norm(x - self.minimizer))

        return {'x': x, 'error': error} if self.minimizer is not None else {'x': x}

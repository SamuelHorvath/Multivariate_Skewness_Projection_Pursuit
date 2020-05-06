import numpy as np
import numpy.linalg as LA

from scipy.linalg import fractional_matrix_power, svd
from scipy.optimize import minimize

from sklearn.datasets import load_iris
from sklearn import preprocessing

import matplotlib.pyplot as plt


class Mardia:
    """
    Multidimensional skewness-based projection pursuit with Mardia (Biometrika, 1970)
    definition of skewness
    b_{1,p} = 1/n^2 sum{i,j}^n [(X_i - bar{X})^T S^{-1} (X_j - bar{X})]^3
    
    If mardia=False, then PCA like approach of Loperfido 
    (https://www.sciencedirect.com/science/article/pii/S0167947317302360)
    is used.
    """

    def __init__(self, X=None, name=None):
        if X is None:
            iris = load_iris()
            X = iris.data
            self.name = 'iris'
        else:
            self.name = name
        X = preprocessing.scale(X)
        self.data = X
        self.S_inv_half = fractional_matrix_power(np.cov(X.T), -0.5)
        self.S_half = LA.inv(self.S_inv_half)
        self.Z = X.dot(self.S_inv_half)

    def run(self, k=1, mardia=True, vis=True):
        c, Z, M_shift, Transf, f_vals = self._get_init()
        print(np.mean(Z.dot(Z.T) ** 3))
        c, f_vals = self.run_opt(c, Z, M_shift, f_vals, mardia)
        if vis:
            self._plot_f_vals(f_vals)

        c_s = [c]

        for l in range(1, k):
            c, Z, M_shift, Transf = self._transfrom_Z(c, Z, M_shift, Transf)
            c, f_vals = self.run_opt(c, Z, M_shift, f_vals, mardia)
            c_s.append(Transf.dot(c))
            if vis:
                self._plot_f_vals(f_vals)

        c_s = np.array(c_s)

        return c_s, f_vals

    def _get_init(self):
        n, p = self.data.shape
        X = self.data

        Z = self.Z

        c_init = self._get_init_c(Z)
        Shift = np.zeros(shape=(n, n))

        Transf = np.eye(p)
        f_vals = [0]

        return c_init, Z, Shift, Transf, f_vals

    def _oneD_objective(self, x, i, c, Z, M_shift, mardia=True):
        c_c = np.copy(c)
        c_c[i] = x
        return self.objective(c_c, Z, M_shift, mardia)

    def _one_iter(self, i, c, Z, M_shift, mardia=True):
        res = minimize(self._oneD_objective, x0=c[i], args=(i, c, Z, M_shift, mardia))
        if res['success']:
            c[i] = res['x']
        c = c / LA.norm(c)
        f_val = -self.objective(c, Z, M_shift)
        return c, f_val

    def run_opt(self, c, Z, M_shift, f_vals, mardia=True, n_iter=None):
        n, p = Z.shape
        if n_iter is None:
            n_iter = 3

        for k in range(n_iter):
            arr = np.arange(p)
            np.random.shuffle(arr)
            for i in arr:
                c, f_val = self._one_iter(i, c, Z, M_shift, mardia)
                f_vals.append(f_val)
            print("{:4d}/{:4d} Iterations, Function value: {:3.4f}"
                  .format(k + 1, n_iter, f_val), end="\r")
        return c, f_vals

    def _transfrom_Z(self, c, Z, Shift, Transf):
        proj = Z.dot(c)
        Shift += np.outer(proj, proj)

        P_c = np.outer(c, c)
        H = Z - Z.dot(P_c)
        _, _, VT = svd(H)

        T = VT[:-1].T
        Z = H.dot(T)

        c_init = self._get_init_c(Z)

        Transf = np.matmul(Transf, T)
        return c_init, Z, Shift, Transf

    @staticmethod
    def _get_init_c(Z):
        n, p = Z.shape
        M_3z = np.zeros(shape=(p ** 2, p))
        for z in Z:
            z_m = z.reshape(p, 1)
            M_3z += np.kron(np.kron(z_m, z_m.T), z_m)

        M_3z /= n

        _, _, VT = svd(M_3z)
        c_init = VT[0]
        return c_init

    @staticmethod
    def objective(c, Z, M_shift, mardia=True):
        c_normalized = c / LA.norm(c)
        proj = Z.dot(c_normalized)
        if mardia:
            V = np.outer(proj, proj) + M_shift
        else:
            V = np.outer(proj, proj)
        f = np.mean(V ** 3)
        return -f

    @staticmethod
    def _plot_f_vals(f_vals):
        plt.plot(f_vals)
        plt.xlabel('Iterations')
        plt.ylabel('Skewness')
        plt.show()

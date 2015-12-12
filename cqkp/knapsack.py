# coding: utf-8

"""A python implementation of some solutions to the knapsack problem.

Copyright (C) 2015, 121onto.  All rights reserved.

Notes
-----
Solvers all solve the following problem:

.. math::

    argmin_{x} \sum_{j=1}^J (a_j - x_j)^2
    s.t. \sum_{j=1}^J x_j \le b
         l \le x_j \le u \forall j


References
----------
M. Patriksson and C. Strömberg. Algorithms for the continuous nonlinear
resource allocation problem – new implementations and numerical studies.
European Journal of Operational Research, 243(3):703–722, 2015.
"""

###########################################################################
## imports

import numpy as np

###########################################################################
## Knapsack solvers

class KnapsackBase(object):
    def __init__(self, J, l=0, u=1, b=1):
        # cache parameters
        self.J = J # M
        self.l = l
        self.u = u
        self.b = b

        # breakpoints
        self.mu_u = None
        self.mu_l = None

    def compute_x_of_mu(self, mu, a):
        self.x[self.mu_l <= mu] = self.l
        self.x[self.mu_u >= mu] = self.u
        where = (self.mu_l > mu) & (self.mu_u < mu)
        self.x[where] = a[where] - (mu / 2.)

    def solve_relaxed(self, a):
        self.x = a + (self.b - a.sum()) / self.J


class KnapsackMB2(KnapsackBase):
    """Median search of breakpoints with 2-sets pegging.
    """
    def __init__(self, J, l=0, u=1, b=1):
        # constraint parameter
        self.bk = b

        # breakpoints
        self.mu_k = None
        self.mu_m = None

        # initialize betas
        self.beta_l = 0.
        self.beta_u = 0.

        # pegging set
        self.N = np.array(range(J), dtype=int)

        # candidate solution
        self.x = np.zeros(J, dtype=np.float64)

        super().__init__(J, l, u, b)

    def solve(self, a):
        # initialize breakpoints
        self.mu_u = 2. * (a - self.u)
        self.mu_l = 2. * (a - self.l)
        self.mu_k = np.hstack([[-10e10], self.mu_l, self.mu_u, [10e10]])

        while True:
            # STEP 1
            if not self.mu_k.size:
                # solve relaxed problem (derived using paper and pencil)
                self.solve_relaxed(a)
                break

            try:
                self.mu_m = np.median(self.mu_k)
            except IndexError:
                print('a:\n{}'.format(a))
                print('mu_k:\n{}'.format(self.mu_k))
                print('mu_k.size:\n{}'.format(self.mu_k.size))
                raise

            self.beta_l = (self.mu_l <= self.mu_m).sum() * self.l
            self.beta_u = (self.mu_u >= self.mu_m).sum() * self.u

            # STEP 2
            self.compute_x_of_mu(self.mu_m, a)
            where = ((self.mu_l > self.mu_m) & (self.mu_u < self.mu_m))
            delta = a[self.N][where[self.N]].sum() + self.beta_u + self.beta_l

            if delta > self.bk:
                # STEP 3.1
                keep = (self.mu_l > self.mu_m)
                self.N = self.N[keep[self.N]]
                self.bk = self.bk - self.beta_l
                self.mu_k = self.mu_k[self.mu_k > self.mu_m]
            elif delta < self.bk:
                # STEP 3.2
                keep = (self.mu_u < self.mu_m)
                self.N = self.N[keep[self.N]]
                self.bk = self.bk - self.beta_u
                self.mu_k = self.mu_k[self.mu_k < self.mu_m]
            else:
                break

        return self.x


class KnapsackMB3(KnapsackBase):
    """Median search of breakpoints with 3-sets pegging.
    """
    def __init__(self, J, l=0, u=1, b=1):
        # constraint parameter
        self.bk = b

        # breakpoints
        self.mu_k = None
        self.mu_m = None

        # initialize betas
        self.beta_l = 0.
        self.beta_u = 0.
        self.mu_lb = -10e10
        self.mu_ub =  10e10

        # pegging sets
        self.N = np.array(range(J), dtype=int)
        self.M = np.array([], dtype=int)

        # candidate solution
        self.x = np.zeros(J, dtype=np.float64)

        super().__init__(J, l, u, b)

    def solve(self, a):
        # initialize breakpoints
        self.mu_u = 2. * (a - self.u)
        self.mu_l = 2. * (a - self.l)
        self.mu_k = np.hstack([self.mu_l, self.mu_u])

        while True:
            # STEP 1
            if not self.mu_k.size:
                # solve relaxed problem (derived using paper and pencil)
                self.solve_relaxed(a)
                break

            self.mu_m = np.median(self.mu_k)
            self.beta_l = (self.mu_l <= self.mu_m).sum() * self.l
            self.beta_u = (self.mu_u >= self.mu_m).sum() * self.u

            # STEP 2
            self.compute_x_of_mu(self.mu_m, a)
            where = ((self.mu_l > self.mu_m) & (self.mu_u < self.mu_m))
            delta = (self.x[self.N][where[self.N]].sum() +
                     self.x[self.M][where[self.M]].sum() +
                     self.beta_u + self.beta_l)

            if delta > self.bk:
                # STEP 3.1
                self.mu_lb = self.mu_m
                keep = (self.mu_l > self.mu_lb)
                self.N = self.N[keep[self.N]]
                keep = ((self.mu_l < self.mu_lb) | (self.mu_l < self.mu_ub) |
                        (self.mu_u > self.mu_lb) | (self.mu_u > self.mu_ub))
                self.N = self.N[keep[self.N]]
                # set logic, ensure no redundant rows
                self.M = self.M[keep[self.M]]
                np.hstack([self.M, np.where(~keep)[0]])

                self.bk = self.bk - self.beta_l
                self.mu_k = self.mu_k[self.mu_k > self.mu_m]
            elif delta < self.bk:
                # STEP 3.2
                self.mu_ub = self.mu_m
                keep = (self.mu_u > self.mu_ub)
                self.N = self.N[keep[self.N]]
                keep = ((self.mu_l < self.mu_lb) | (self.mu_l < self.mu_ub) |
                        (self.mu_u > self.mu_lb) | (self.mu_u > self.mu_ub))
                self.N = self.N[keep[self.N]]
                # set logic, ensure no redundant rows
                self.M = self.M[keep[self.M]]
                np.hstack([self.M, np.where(~keep)[0]])

                self.bk = self.bk - self.beta_u
                self.mu_k = self.mu_k[self.mu_k < self.mu_m]
            else:
                break

        return self.x


###########################################################################
## tests

def test_mb2(prng=None, verbose=False, tol=1e-6):
    for J in range(1, 10):
        J = int(1.5 ** J)
        a = prng.rand(J)
        mb2 = KnapsackMB2(J=J, l=0, u=1, b=1)

        def fun(x):
            return np.linalg.norm(a - x) ** 2.

        def jac(x):
            return 2. * (x - a)

        x_star_mb2 = mb2.solve(a)
        x_star_slsqp = optimize.minimize(
            fun,
            prng.uniform(-1,1,J),
            method='SLSQP',
            jac=jac,
            bounds=[(0,1) for i in range(a.size)],
            constraints=dict(
                type='eq',
                fun = lambda x: x.sum() - 1.,
                jac = lambda x: np.ones(x.size, dtype=int)
            )
        ).x

        if verbose:
            print('-' * 40)
            print('Solutions for J = {}'.format(J))
            print('MB2 found: {}'.format(fun(x_star_mb2)))
            print('SLSQP found: {}'.format(fun(x_star_slsqp)))

        assert(fun(x_star_mb2) <= fun(x_star_slsqp) + tol)

def test_mb3(prng=None, verbose=False, tol=1e-6):
    for J in range(1, 10):
        J = int(1.5 ** J)
        a = prng.rand(J)
        mb3 = KnapsackMB3(J=J, l=0, u=1, b=1)

        def fun(x):
            return np.linalg.norm(a - x) ** 2.

        def jac(x):
            return 2. * (x - a)

        x_star_mb3 = mb3.solve(a)
        x_star_slsqp = optimize.minimize(
            fun,
            prng.uniform(-1,1,J),
            method='SLSQP',
            jac=jac,
            bounds=[(0,1) for i in range(a.size)],
            constraints=dict(
                type='eq',
                fun = lambda x: x.sum() - 1.,
                jac = lambda x: np.ones(x.size, dtype=int)
            )
        ).x

        if verbose:
            print('-' * 40)
            print('Solutions for J = {}'.format(J))
            print('MB3 found: {}'.format(fun(x_star_mb3)))
            print('SLSQP found: {}'.format(fun(x_star_slsqp)))

        assert(fun(x_star_mb3) <= fun(x_star_slsqp) + tol)

###########################################################################
## main

if __name__ == '__main__':
    # imports for tests
    import scipy.optimize as optimize
    try:
        from config import SEED
    except:
        SEED = 1234

    # includes for tests go here
    prng = np.random.RandomState(SEED)
    test_mb2(prng, True)
    test_mb3(prng, True)

    print('\n\n--> All tests ran without error.\n\n')

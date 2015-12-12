# A DC optimization algorithm

This code is based on the following paper:

P. D. Tao and L. T. H. An. A D.C. optimization algorithm for solving the trust-region subproblem. *SIAM Journal on Optimization*, 8(2):476–505, 1998.

Built and tested using python 3.5.

Suggestions, bug reports, and improvements are welcome.

**Notes**: code is an alpha release and is not thoroughly tested.  Existing tests use scipy.optimize.minimize and may fail when SLSQP finds a local minima.

import numpy as np


def gll_nodes(npol):

    # Get machine epsilon
    eps = np.finfo(np.float64).eps

    # Number of nodes
    ngll = npol + 1

    # Chebychev-Gauss-Lobatto nodes as first guess
    x = -np.cos(np.pi*np.arange(0, npol+1)/npol)

    # Vandermonde matrix
    P = np.zeros((ngll, ngll))

    # Use recursion
    xold = 2. * np.ones(ngll)

    while np.max(np.abs(x-xold)) > eps:
        xold[:] = x[:]
        P[:, 0] = np.ones(ngll)
        P[:, 1] = x[:]
        for k in range(1, npol):
            P[:, k+1] = ((2*(k+1)-1) * x[:] * P[:, k]-(k)*P[:, k-1]) / (k+1)

        x[:] = xold[:] - (x[:] * P[:, ngll-1] - P[:, npol-1]) / \
            (ngll * P[:, ngll-1])

    w = 2. / (npol * ngll * P[:, ngll-1]**2)

    return (x, w, P)


def lagrange_any(x, xigll, npol):
    _x = np.array([x])
    return lagrange_polynomials(_x, xigll, npol)[0], \
        lagrange_polynomials_first_derivative(_x, xigll, npol)[0]


def lagrange_polynomials(x, xigll, npol):
    nx = len(x)
    lp = np.zeros((nx, npol+1))
    for i in range(npol+1):
        prod = 1.
        for k in range(npol+1):
            if (k != i):
                num = x - xigll[k]
                den = xigll[i] - xigll[k]
                prod *= num / den
        # end for k
        lp[:, i] = prod
    return lp


def lagrange_polynomials_first_derivative(x, xigll, npol):
    nx = len(x)
    lp = np.zeros((nx, npol+1))
    for i in range(npol+1):
        num = 0.
        den = 1.
        for k in range(npol+1):
            if (k != i):
                # Compute numerator
                prod = 1.
                for l in range(npol+1):
                    if (l != k) and (l != i):
                        prod *= x - xigll[l]

                num += prod
                # Compute denominator
                den *= xigll[i]-xigll[k]

        lp[:, i] = num / den

    return lp

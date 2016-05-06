# Numerical differentiator from data -- gives approximation to the derivative
# of a function which is provided by data points
#
# Written by Jonathan D. Slavin (jslavin@cfa.harvard.edu) 6 May 2016

import numpy as np
from scipy.special import comb

def coeffs(M):
    """
    Generate the "Smooth noise-robust differentiators" as defined in Pavel
    Holoborodko's formula for c_k

    Parameters
    ----------
    M : int
        the order of the differentiator

    c : float array of length M
        coefficents for k = 1 to M
    """
    m = (2*M - 2)/2
    k = np.arange(1, M+1)
    c = 1./2.**(2*m + 1)*(comb(2*m, m - k + 1) - comb(2*m, m - k - 1))
    return c

def holo_diff(x,y,M=2):
    """
    Implementation of Pavel Holoborodko's method of "Smooth noise-robust
    differentiators" see
    http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/
    smooth-low-noise-differentiators
    Creates a numerical approximation to the first derivative of a function
    defined by data points.  End point approximations are found from
    approximations of lower order.  Greater smoothing is achieved by using a
    larger value for the order parameter, M.

    Parameters
    ----------
    x : float array or scalar
        abscissa values of function or, if scalar, uniform step size
    y : float array
        ordinate values of function (same length as x if x is an array)
    M : int, optional (default = 2)
        order for the differentiator - will use surrounding 2*M + 1 points in
        creating the approximation to the derivative

    Returns
    -------
    dydx : float array
        numerical derivative of the function of same size as y
    """
    if np.isscalar(x):
        x = x*np.arange(len(y))
    assert len(x) == len(y), 'x and y must have the same length if x is ' + \
            'an array, len(x) = {}, len(y) = {}'.format(len(x),len(y))
    N = 2*M + 1
    m = (N - 3)/2
    c = coeffs(M)
    df = np.zeros_like(y)
    nf = len(y)
    fk = np.zeros((M,(nf - 2*M)))
    for i,cc in enumerate(c):
        # k runs from 1 to M
        k = i + 1
        ill = M - k
        ilr = M + k
        iul = -M - k
        # this formulation is needed for the case the k = M, where the desired
        # index is the last one -- but range must be given as [-2*M:None] to
        # include that last point
        iur = ((-M + k) or None)
        fk[i,:] = 2.*k*cc*(y[ilr:iur] - y[ill:iul])/(x[ilr:iur] - 
                x[ill:iul])
    df[M:-M] = fk.sum(axis=0)
    # may want to incorporate a variety of methods for getting edge values,
    # e.g. setting them to 0 or just using closest value with M of the ends.
    # For now we recursively calculate values closer to the edge with
    # progressively lower order approximations -- which is in some sense
    # ideal, though maybe not for all cases
    if M > 1:
        dflo = holo_diff(x[:2*M],y[:2*M],M=M-1)
        dfhi = holo_diff(x[-2*M:],y[-2*M:],M=M-1)
        df[:M] = dflo[:M]
        df[-M:] = dfhi[-M:]
    else:
        df[0] = (y[1] - y[0])/(x[1] - x[0])
        df[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    return df

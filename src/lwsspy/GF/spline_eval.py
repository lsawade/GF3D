import numpy as np
from .constants import R_PLANET_KM


def spline_evaluation(
        xpoint: np.ndarray, ypoint: np.ndarray, spline_coefficients: np.ndarray,
        x_evaluate_spline: float):

    # number of input points and coordinates of the input points
    npoint = len(xpoint)

    # initialize to the whole interval
    index_lower = 0
    index_higher = npoint - 1

    # determine the right interval to use, by dichotomy
    while (index_higher - index_lower > 1):

        # compute the middle of the interval
        index_loop = int((index_higher + index_lower) / 2)

        if (index_loop < 1):
            index_loop = 1

        if (xpoint[index_loop] > x_evaluate_spline):

            index_higher = index_loop
        else:
            index_lower = index_loop

    # test that the interval obtained does not have a size of zero
    # (this could happen for instance in the case of duplicates in the input list of points)
    if (xpoint[index_higher] == xpoint[index_lower]):
        print('Error: invalid spline evalution index_higher/index_lower = ',
              index_higher, index_lower, 'range = ', 1, npoint)
        print(
            '       point x = ', x_evaluate_spline,
            ' x radius = ', x_evaluate_spline * R_PLANET_KM,  '(km)')
        raise ValueError('incorrect interval found in spline evaluation')

    coef1 = (xpoint[index_higher] - x_evaluate_spline) / \
        (xpoint[index_higher] - xpoint[index_lower])
    coef2 = (x_evaluate_spline - xpoint[index_lower]) / \
        (xpoint[index_higher] - xpoint[index_lower])

    y_spline_obtained = \
        coef1*ypoint[index_lower] + coef2*ypoint[index_higher] \
        + ((coef1**3 - coef1)*spline_coefficients[index_lower]
           + (coef2**3 - coef2)*spline_coefficients[index_higher]) \
        * ((xpoint[index_higher] - xpoint[index_lower])**2)/6.0

    return y_spline_obtained

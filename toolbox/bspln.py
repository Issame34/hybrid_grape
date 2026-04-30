import numpy as np
from scipy.interpolate import BSpline


def bknots_on_interval(t_left, t_right, n, k):
    """
    Create B-spline knots uniformly distributed over the interval (t_left, t_right)
    """
    knots_left = np.full( shape=k, fill_value=t_left )
    knots_right = np.full( shape=k, fill_value=t_right )
    knots_mid = np.linspace( t_left, t_right, n-k+1 )
    knots_glob = np.concatenate( [knots_left, knots_mid, knots_right] )
    return knots_glob


def setup_bspline_builder(time_start, time_end, n, k, skip_left=0, skip_right=0):
    """
    Setup B-spline builder on different time grids with other parameters fixed.
    """ 
    knots = bknots_on_interval(time_start, time_end, n, k)
    
    def bspline_builder(time_grid):
        
        funcs_on_grid = np.zeros(
            shape=(n-skip_left-skip_right, time_grid.shape[0]),
            dtype=time_grid.dtype
        )
        for i in range(skip_left, n-skip_right):

            knots_internal = knots[ i : i+k+2 ]
            bfunc = BSpline.basis_element( knots_internal )

            where_nonzero = (time_grid > knots_internal[0]) & (time_grid < knots_internal[-1])
            t_where_nonzero = time_grid[ where_nonzero ]

            funcs_on_grid[ i-skip_left, where_nonzero ] = bfunc(t_where_nonzero)

            if time_grid[0] == time_start and skip_left == 0:
                funcs_on_grid[0, 0] = 1.0

            if time_grid[-1] == time_end and skip_right == 0:
                funcs_on_grid[-1, -1] = 1.0
                
        return funcs_on_grid

    return bspline_builder
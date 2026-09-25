import numpy as np
from numba import njit, float64


@njit(cache=True)
def fast_cheb_eval(norm, coeffs):
    """
    norm: (N,) normalized wavelength/position values
    coeffs: (Deg, N) coefficients for EACH point

    cheb on point x_norm[i] = coeffs[0,i]*T0(x_norm[i]) + coeffs[1,i]*T1(x_norm[i]) + ...

    T0(x) = 1
    T1(x) = x
    Tn(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)

    https://en.wikipedia.org/wiki/Chebyshev_polynomials

    We use https://en.wikipedia.org/wiki/Clenshaw_algorithm for evaluation (see numpy)

    """
    n_samples = norm.shape[0]
    n_deg = coeffs.shape[0]
    results = np.empty(n_samples, dtype=float64)

    for i in range(n_samples):   # This can be made parallel using prange I think
        v = norm[i]
        # Clenshaw recurrence
        if n_deg == 1:
            results[i] = coeffs[0, i]
        else:
            sv = 2.0 * v
            d2 = 0.0
            d1 = 0.0
            for j in range(n_deg - 1, 0, -1):
                tmp = d1
                d1 = coeffs[j, i] + sv * d1 - d2
                d2 = tmp
            results[i] = coeffs[0, i] + v * d1 - d2

    return results


@njit(cache=True)
def fast_cheb_coeffs_kernel(x_norm, y_norm, model_matrices):
    """
    x_norm: (N,) normalized x-coordinates
    y_norm: (N,) normalized y-coordinates

    model_matrices: (M, DegX, DegY) model matrices. DegX and DegY are usually 3 or 5 and they are equal.
    model_metrices contains M matrices of coeefficients.

    The output is (M, N) where each row corresponds to the evaluated coefficients for each model at the given x and y.

        The algorithm is a two-step Clenshaw evaluation (see fast_cheb_eval above):
    """
    n_models, deg_x, deg_y = model_matrices.shape
    n_samples = x_norm.shape[0]
    out = np.empty((n_models, n_samples))

    for i in range(n_samples):      # This can be made parallel using prange I think
        vx = x_norm[i]
        vy = y_norm[i]

        for m in range(n_models):
            matrix = model_matrices[m]

            # we evaluate the y-dimension for each x-degree
            # We want to reduce the matrix[deg_x, deg_y] to a 1D array of size deg_x
            y_reduced = np.zeros(deg_x)

            # I am evaluating the result not as Sum_i Sum_j c_ij * T_i(x) * T_j(y) but as Sum_i (c_i0 + c_i1*T1(y) + c_i2*T2(y) + ...)*T_i(x)
            # Here I evaluate the (c_i0 + c_i1*T1(y) + c_i2*T2(y) + ...) part with the same algorithm as above
            for r in range(deg_x):
                d1, d2 = 0.0, 0.0
                sv_y = 2.0 * vy
                for c in range(deg_y - 1, 0, -1):
                    tmp = d1
                    d1 = matrix[r, c] + sv_y * d1 - d2
                    d2 = tmp
                y_reduced[r] = matrix[r, 0] + vy * d1 - d2

            # Here I evaluate the Sum_i T_i(x) * ...  part with the same algorithm as above.
            # At the end you can think of y_reduced as "new coefficients" and the exercise is to evaluate cheb on x with the y_reduced coefficients.
            d1_x, d2_x = 0.0, 0.0
            sv_x = 2.0 * vx
            for r in range(deg_x - 1, 0, -1):
                tmp = d1_x
                d1_x = y_reduced[r] + sv_x * d1_x - d2_x
                d2_x = tmp
            out[m, i] = y_reduced[0] + vx * d1_x - d2_x

    return out


@njit(fastmath=True, cache=True)
def fast_polyval2d(x, y, m):
    """Vectorized 2D power series evaluation: sum(m[i,j] * x^i * y^j)

    The algorithm (Horner) is taken from the numpy implementation, that is - instead of
    increasing the power of the x and y. I start from the highest degree and loop backwards.
    e.g.  Ay^2+By+C, is calculated as ((A⋅y)+B)⋅y+C.

    See https://github.com/numpy/numpy/blob/main/numpy/polynomial/polynomial.py

    """
    n_samples = x.shape[0]
    deg_x, deg_y = m.shape
    out = np.empty(n_samples)

    for i in range(n_samples):   # eventually prange
        val = 0.0
        # Horner's method or simple power accumulation
        # We evaluate y first
        for r in range(deg_x):
            y_part = 0.0
            # Evaluate the y-polynomial
            for c in range(deg_y - 1, -1, -1):
                y_part = m[r, c] + y[i] * y_part

            # Multiply by x^r and add to total  ---> instead of doing c_ij * x^i * y^j we do (c_i0 + c_i1*y + c_i2*y^2 + ...)*x^i
            val += y_part * (x[i]**r)
        out[i] = val
    return out

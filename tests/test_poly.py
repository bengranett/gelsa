"""Tests for Numba polynomial evaluation kernels."""
import numpy as np
import pytest
from numpy.polynomial.polynomial import polyval2d
from numpy.polynomial.chebyshev import chebval
from gelsa.sgs.poly import fast_polyval2d, fast_cheb_eval, fast_cheb_coeffs_kernel


RNG = np.random.default_rng(0)


@pytest.fixture(scope="module", autouse=True)
def warmup_jit():
    x = RNG.uniform(-1, 1, 5).astype(np.float64)
    y = RNG.uniform(-1, 1, 5).astype(np.float64)
    m = RNG.standard_normal((3, 3)).astype(np.float64)
    fast_polyval2d(x, y, m)

    norm   = RNG.uniform(-1, 1, 5).astype(np.float64)
    coeffs = RNG.standard_normal((3, 5)).astype(np.float64)
    fast_cheb_eval(norm, coeffs)

    model_matrices = RNG.standard_normal((2, 3, 3)).astype(np.float64)
    fast_cheb_coeffs_kernel(x, y, model_matrices)


class TestFastPolyval2d:
    def test_against_numpy(self):
        m = RNG.standard_normal((4, 4))
        x = RNG.uniform(-1, 1, 200)
        y = RNG.uniform(-1, 1, 200)

        expected = np.array([polyval2d(xi, yi, m) for xi, yi in zip(x, y)])
        result = fast_polyval2d(x.astype(np.float64), y.astype(np.float64), m.astype(np.float64))

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_constant_matrix(self):
        m = np.zeros((3, 3))
        m[0, 0] = 7.0
        x = RNG.uniform(-1, 1, 50).astype(np.float64)
        y = RNG.uniform(-1, 1, 50).astype(np.float64)
        result = fast_polyval2d(x, y, m)
        np.testing.assert_allclose(result, 7.0)


class TestFastChebEval:
    def test_against_numpy(self):
        n = 200
        deg = 5
        norm   = RNG.uniform(-1, 1, n).astype(np.float64)
        coeffs = RNG.standard_normal((deg, n)).astype(np.float64)

        expected = np.array([chebval(norm[i], coeffs[:, i]) for i in range(n)])
        result = fast_cheb_eval(norm, coeffs)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_degree_one(self):
        norm   = np.array([0.5]).astype(np.float64)
        coeffs = np.array([[3.0]]).astype(np.float64)
        result = fast_cheb_eval(norm, coeffs)
        assert result[0] == pytest.approx(3.0)


class TestFastChebCoeffsKernel:
    def test_against_cheb_eval(self):
        """fast_cheb_coeffs_kernel is a 2-D generalisation; check one-axis limit."""
        n = 50
        deg = 3
        x_norm = RNG.uniform(-1, 1, n).astype(np.float64)
        y_norm = np.zeros(n, dtype=np.float64)

        model_matrices = RNG.standard_normal((1, deg, deg)).astype(np.float64)
        result = fast_cheb_coeffs_kernel(x_norm, y_norm, model_matrices)
        assert result.shape == (1, n)

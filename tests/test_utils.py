import numpy as np
import pytest
from gelsa.utils import ensurelist, asarray, is_number, intrange


class TestEnsurelist:
    def test_scalar_int(self):
        assert ensurelist(5) == [5]

    def test_scalar_float(self):
        assert ensurelist(3.14) == [3.14]

    def test_string_stays_wrapped(self):
        assert ensurelist("hello") == ["hello"]

    def test_list_passthrough(self):
        x = [1, 2, 3]
        assert ensurelist(x) is x

    def test_array_passthrough(self):
        x = np.array([1, 2, 3])
        result = ensurelist(x)
        assert result is x


class TestAsarray:
    def test_scalar(self):
        result = asarray(5)
        assert isinstance(result, np.ndarray)
        assert result[0] == 5

    def test_list(self):
        result = asarray([1, 2, 3])
        np.testing.assert_array_equal(result, [1, 2, 3])


class TestIsNumber:
    def test_integer_string(self):
        assert is_number("42") is True

    def test_float_string(self):
        assert is_number("3.14") is True

    def test_plain_string(self):
        assert is_number("hello") is False

    def test_none(self):
        assert is_number(None) is False

    def test_numeric(self):
        assert is_number(7) is True


class TestIntrange:
    def test_length_at_least_limit(self):
        result = intrange(0, 1, 0.5, limit=3)
        assert len(result) >= 3

    def test_endpoints(self):
        result = intrange(0.0, 10.0, 1.0)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(10.0)

    def test_reversed_order(self):
        result_fwd = intrange(0, 10, 1)
        result_rev = intrange(10, 0, 1)
        np.testing.assert_allclose(result_fwd, result_rev)

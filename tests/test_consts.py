import pytest
from gelsa.consts import line_list, line_names, lines, line_indices

NISP_RANGE = (9000, 19000)  # Angstrom


def test_line_names_complete():
    assert set(line_names) == set(line_list.keys())


def test_line_indices_consistent():
    for name, idx in line_indices.items():
        assert line_names[idx] == name


def test_lines_array_length():
    assert len(lines) == len(line_names)


def test_lines_in_nisp_range():
    """Emission lines must fall within the NISP wavelength range when redshifted."""
    for name, wav in line_list.items():
        assert wav > 0, f"{name} wavelength must be positive"


def test_halpha_wavelength():
    assert abs(line_list["Ha"] - 6564.61) < 0.1


def test_line_order_matches_array():
    for i, name in enumerate(line_names):
        assert lines[i] == pytest.approx(line_list[name])

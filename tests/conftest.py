import os
import pytest


CALIB_CONFIG = os.path.abspath("../gelsa-spectra/calib/gelsa_config.json")
CALIB_DIR    = os.path.abspath("../gelsa-spectra/calib")

needs_calib = pytest.mark.skipif(
    not os.path.exists(CALIB_CONFIG),
    reason="calibration files not available",
)


@pytest.fixture(scope="session")
def gelsa_frame():
    """A Gelsa instance and a SpecFrame for integration tests."""
    import gelsa
    G = gelsa.Gelsa(config_file=CALIB_CONFIG, calibdir=CALIB_DIR)
    frame = G.new_frame(10, 0, 0)
    return G, frame

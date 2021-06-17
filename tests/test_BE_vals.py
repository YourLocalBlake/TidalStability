from pytest import approx
import sys
import os

from tidal_stability.utils import (
    get_BE_equilibrium_radius,
    get_BE_mass_0to5sqrt5o16,
    get_BE_mass_1to4_ratio,
)


def blockprint():  # This will disable printing to console as many functions print.
    sys.stdout = open(os.devnull, "w")


blockprint()

# These vals should eventually be moved into like a big file that can be called?
BE_MASS_1TO4_1 = get_BE_mass_1to4_ratio(1)
BE_MASS_1TO4_09 = get_BE_mass_1to4_ratio(0.9)
BE_MASS_1TO4_01 = get_BE_mass_1to4_ratio(0.1)
BE_MASS_1TO4_001 = get_BE_mass_1to4_ratio(0.001)

BE_MASS_0S5_1 = get_BE_mass_0to5sqrt5o16(1)
BE_MASS_0S5_1e12 = get_BE_mass_0to5sqrt5o16(1e12)
BE_MASS_0S5_I09 = get_BE_mass_0to5sqrt5o16(BE_MASS_1TO4_09)  # I for input, meaning we
# are using input from another function.
BE_MASS_0S5_I01 = get_BE_mass_0to5sqrt5o16(BE_MASS_1TO4_01)

BE_EQU_RAD_I09 = get_BE_equilibrium_radius(BE_MASS_0S5_I09)
BE_EQU_RAD_I01 = get_BE_equilibrium_radius(BE_MASS_0S5_I01)


class TestClass:
    def test_BE(self):
        assert BE_MASS_1TO4_1 == approx(4)
        assert BE_MASS_1TO4_09 == approx(2.472434740940327)
        assert BE_MASS_1TO4_01 == approx(1.118131307246284)
        assert BE_MASS_1TO4_001 == approx(1.004754680256626)
        assert BE_MASS_0S5_1 == 0.0
        assert BE_MASS_0S5_1e12 == 5 / 16 * 5 ** (1 / 2)
        assert BE_MASS_0S5_I09 == approx(0.628894118671815)
        assert BE_MASS_0S5_I01 == approx(
            0.06987712429692489
        )  # Not approx 5/16 * sqrt(5)
        assert BE_EQU_RAD_I09[0] == approx(0.4334797370040493)
        assert BE_EQU_RAD_I09[1] == approx(0.6336034966348003)
        assert BE_EQU_RAD_I01[0] == approx(0.0419706813343878)
        assert BE_EQU_RAD_I01[1] == approx(0.3968387491412735)

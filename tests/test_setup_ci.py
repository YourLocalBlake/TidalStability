from pytest import approx

from tidal_stability.solve.geometry import get_Ax, get_Ay, get_Az
from tidal_stability.data_formats import *
from tidal_stability.solve import *


def test_trivial():
    # This is simply to have a function which is called
    return 1


# Basic class and trivially checking a few known values.
class TestClass:
    def test_ellip_ints(self):
        assert get_Ax(x=0.5, y=0.6, z=0.7) == approx(3.845896270355620)
        assert get_Ay(x=0.5, y=0.6, z=0.7) == approx(3.108621338096264)
        assert get_Az(x=0.5, y=0.6, z=0.7) == approx(2.569291915357639)

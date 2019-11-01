"""
Constants related to the orientation and structure of the ellipsoid
"""
from mpmath import elliprd


def get_Ax(*, x, y, z):
    """
    Returns the degenerate Carlson symmetric elliptic integral of the third kind. Note that the permutation of the first
    two arguments does not change the result
    """
    return float(2/3 * (elliprd(y**2, z**2, x**2)))


def get_Ay(*, x, y, z):
    """
    Returns the degenerate Carlson symmetric elliptic integral of the third kind. Note that the permutation of the first
    two arguments does not change the result
    """
    return float(2/3 * (elliprd(x**2, z**2, y**2)))


def get_Az(*, x, y, z):
    """
    Returns the degenerate Carlson symmetric elliptic integral of the third kind. Note that the permutation of the first
    two arguments does not change the result
    """
    return float(2/3 * (elliprd(x**2, y**2, z**2)))

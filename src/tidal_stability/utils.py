"""
Functions and classes that can be useful.
"""
from enum import IntEnum
from math import sqrt
from mpmath import findroot, mp
from numpy import linspace
from scipy.optimize import brenth

from tidal_stability.solve.geometry import get_Ax, get_Ay, get_Az


class ODEIndex(IntEnum):
    """
    Enumerated number for array index for variables in the ODEs
    """

    # Dimensionful units
    a1 = 0
    a1dot = 1
    a2 = 2
    a2dot = 3
    a3 = 4
    a3dot = 5
    θ = 6
    θdot = 7
    ϕ = 8
    ϕdot = 9

    # Dimensionless unit
    x = 0
    xdot = 1
    y = 2
    ydot = 3
    z = 4
    zdot = 5
    # Theta, phi are the same as above.


class EllipIndex(IntEnum):
    """
    Enumerated number for array index for variables in the elliptical integrals
    """

    x = 0
    y = 1
    z = 2
    a1 = 0
    a2 = 1
    a3 = 2


def get_BE_equilibrium_radius(mass_r):
    """
    :param mass_r, int, goes between 0 and 5 sqrt(5)/16
    Obtain the two radius solutions to the equilibrium configuration for a BE mass
    """

    def _eq_rad_dimless(x):
        """
        Calculates the radius for equilibrium.
        RHS of equation where if LHS = 0 then cloud is in equilibirum
        Citation: B.Draine, Physics of the interstellar and intergalactic medium (Princeton University Press,
                  Oxfordshire, 2011
        """
        return x ** 4 - x * mass_r + 3 / 5 * mass_r ** 2

    if mass_r > (5 * sqrt(5)) / 16:
        print(
            "Selected to use a mass ratio greater than the stable BE mass, returning the local minimum value"
        )
        val_range = linspace(0, 1.5, 2000000)
        val_min, val_idx = min(
            (val, idx)
            for (idx, val) in enumerate([_eq_rad_dimless(i) for i in val_range])
        )
        return val_min, val_min

    val_range = linspace(0, 1, 1000000)
    val_min, val_idx = min(
        (val, idx) for (idx, val) in enumerate([_eq_rad_dimless(i) for i in val_range])
    )
    mp.dps = 100  # Decimal points to use with mp math.

    try:
        # rad_low = findroot(_eq_rad_dimless, val_range[0], solver="muller")
        rad_low = brenth(_eq_rad_dimless, 0, val_range[val_idx] - 1e-10)
    except ValueError:
        rad_low = findroot(_eq_rad_dimless, val_range[0], solver="muller")
        print("ONLY ONE ROOT EXISTS!")
        return float(rad_low), float(rad_low)
    try:
        rad_hig = brenth(_eq_rad_dimless, val_range[val_idx] + 1e-10, val_range[-1])
        print(
            "and at this pressure an equilibrium radii of r = {}, {} cm exists.".format(
                round(rad_low, 10), round(rad_hig, 10)
            )
        )
        print(
            "MAKE SURE THE RADII ARE DIFFERENT.",
            "\n",
            "The root-finding algorithm CAN BE temperamental for the larger root",
        )
    except ValueError:
        print(
            "Root finding algorithm could not find a root to the radius equation."
            "Did you give a proper initial guess?, try changing initial guess, else no equilibrium might exist"
        )
        print(
            "You might want to check which root is failing. The root-finding algorithm is temperamental for the "
            "larger root. Try using more decimal points."
        )
        rad_low = -1
        rad_hig = -1

    del val_range
    return rad_low, rad_hig


def get_BE_mass_1to4_ratio(percent_be_mass):
    """
    :param percent_be_mass: int. between 0 and 1. The ratio for how massive the cloud should be
    :return int. value between 1-4 representing the mass of the Bonnor-Ebert Sphere.
    """

    def eq(x):
        val = (
            percent_be_mass
            - 1 / (5 * sqrt(5) / 16) * x * sqrt((x - 1) / (3 / 5 * x ** 2)) ** 3
        )
        return val

    if 0 < percent_be_mass <= 1:
        mass_r = brenth(eq, 1, 4)

    elif percent_be_mass > 1:
        print("You've selected an unstable cloud, Manually set ρ_real_over_ρ_pressure")
        mass_r = -1

    elif percent_be_mass < 0:
        print("Negative mass ratio selected")
        mass_r = -1
    else:
        print(
            "Failed to calculate cloud mass, did you input a interger between 0 and 1? I got {}".format(
                percent_be_mass
            )
        )
        mass_r = -1

    return mass_r


def get_BE_mass_0to5sqrt5o16(ρ_normalised, override_percentage=-1):
    """
    Calculate the value between 0 and 5 sqrt(5)/16 for density
    Requires normalised density and returns the mass_ratio required. THe 5sqrt(5)/16 is NOT taken into account
    """
    # todo: far future, fix the printed states so it says 0 to 100% mBE
    if ρ_normalised < 1:
        print("Rho < 1 will result in zero mass specify override percentage")
        return override_percentage
    if ρ_normalised > 4:
        print(
            "Selected a mass over the maximum mass. Returning maximum value of 5 * sqrt(5)/16. Manually insert value"
        )
        return 5 * sqrt(5) / 16
    if ρ_normalised == 4:
        print("Maximum stable mass selected")
        return 5 * sqrt(5) / 16
    else:
        print(
            "Solving for a mass cloud of "
            + str(
                ρ_normalised
                * sqrt((ρ_normalised - 1) / (3 / 5 * ρ_normalised ** 2)) ** 3
            )
            + " m_BE"
        )
        return (
            ρ_normalised * sqrt((ρ_normalised - 1) / (3 / 5 * ρ_normalised ** 2)) ** 3
        )


def axis_length_ratio_solver(start, stop):
    """
    Generate the length_2/length_1 axis ratio based on the length_3/length_1 axis ratio which is fed in.
    I.E generate the x/y axis ratio based on the z/x value
    :param start: int, the value which we start solving at
    :param stop: int, the value which we stop solving at
    :return: list of lists: the axis ratio lengths and Chandrasekhar values. Scaled as a_i/R^{(1/3)}
    """
    from scipy.optimize import brenth

    def equ_eq(length1, length2):
        """
        :param length1: int, the value that will be solved for a2/a1
        :param length2: int, the known value, a3/a1
        :return: int, the ratio length_2/length_1 consistent with length_3/length_1
        :return:
        """
        Ax = get_Ax(x=1, y=length1, z=length2)
        Ay = get_Ay(x=1, y=length1, z=length2)
        Az = get_Az(x=1, y=length1, z=length2)
        return (
            (Ax - Az * length2 ** 2) / (Ay * length1 ** 2 - Az * length2 ** 2)
            - 3 / length2 ** 2
            - 1
        )

    vals = linspace(
        start, stop, 1000
    )  # This is the range of a3/a1 values we will feed to the functions
    sols = []  # This is a2/a1 values when using a given a3/a1

    for val in vals:
        # I found brenthh to be the fastest solver.
        sols.append(brenth(equ_eq, val + 0.0001, 0.9999, args=(val,)))

    plot_vals_a1 = [
        1 / (sols[i] * vals[i]) ** (1 / 3) for i in range(len(sols))
    ]  # This gives a1/(a1 a2 a3)^3
    plot_vals_a2 = [
        1 / (vals[i] / sols[i] ** 2) ** (1 / 3) for i in range(len(sols))
    ]  # This gives a2/(a1 a2 a3)^3
    plot_vals_a3 = [
        1 / (sols[i] / vals[i] ** 2) ** (1 / 3) for i in range(len(sols))
    ]  # This gives a3/(a1 a2 a3)^3

    # The following values are for p=0 in Chandrasekhar.
    chan_a3a1 = [
        0.91355,
        0.80902,
        0.66913,
        0.54464,
        0.50000,
        0.48481,
        0.46947,
        0.45399,
        0.40674,
        0.32557,
        0.30902,
        0.25882,
        0.19081,
    ]
    # chan_a2a1 = [0.93188, 0.84112, 0.70687, 0.57787, 0.53013, 0.51373, 0.49714, 0.48040, 0.42898, 0.34052, 0.32254,
    #              0.26827, 0.19569]

    chan_a1 = [
        1.0551,
        1.1369,
        1.2835,
        1.4701,
        1.5567,
        1.5894,
        1.6242,
        1.6613,
        1.7896,
        2.0816,
        2.1568,
        2.4330,
        2.9919,
    ]
    chan_a2 = [
        0.9832,
        0.9563,
        0.9072,
        0.8495,
        0.8253,
        0.8165,
        0.8074,
        0.7981,
        0.7677,
        0.7088,
        0.6957,
        0.6527,
        0.5855,
    ]
    chan_a3 = [
        0.9639,
        0.9198,
        0.8588,
        0.8007,
        0.7784,
        0.7706,
        0.7625,
        0.7542,
        0.7279,
        0.6777,
        0.6665,
        0.6297,
        0.5709,
    ]

    return (
        vals,
        sols,
        [plot_vals_a1, plot_vals_a2, plot_vals_a3],
        [chan_a3a1, chan_a1, chan_a2, chan_a3],
    )

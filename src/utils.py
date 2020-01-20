"""
Functions and classes that can be useful.
"""
from enum import IntEnum
from math import sqrt, pi
from mpmath import findroot, mp
from numpy import linspace, arange
from scipy.optimize import brenth

from src.solve.geometry import get_Ax, get_Ay, get_Az


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
    Obtain the two radius solutions to the equilibrium configuration for a BE mass
    :param mass_r, int, goes between 0 and 5 sqrt(5)/16
    used to be Get_BE_Vals[1][0]
    """

    def _eq_rad_dimless(x):
        """
        Calculates the radius for equilibrium.
        RHS of equation where if LHS = 0 then cloud is in equilibirum
        Citation: B.Draine, Physics of the interstellar and intergalactic medium (Princeton University Press,
                  Oxfordshire, 2011
        """
        return x**4 - x * mass_r + 3/5 * mass_r**2

    if mass_r > (5 * sqrt(5))/16:
        print("Selected to use a mass ratio greater than the stable BE mass, returning the local minimum value")
        val_range = linspace(0, 1.5, 2000000)
        val_min, val_idx = min((val, idx) for (idx, val) in enumerate([_eq_rad_dimless(i) for i in val_range]))
        return val_min, val_min

    val_range = linspace(0, 1, 1000000)
    val_min, val_idx = min((val, idx) for (idx, val) in enumerate([_eq_rad_dimless(i) for i in val_range]))
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
        print("and at this pressure equilibrium radii of r = {} and {} cm exist.".format(round(rad_low, 10),
                                                                                          round(rad_hig, 10)),
              "\n" + "The root-finding algorithm can temperamental for the larger root, so check they are different")
    except ValueError:
        print("Root finding algorithm could not find a root to the radius equation."
              "Did you give a proper initial guess?, try changing initial guess, else no equilibrium might exist")
        print("You might want to check which root is failing. The root-finding algorithm is temperamental for the "
              "larger root. Try using more decimal points.")
        rad_low = -1
        rad_hig = -1

    del val_range
    return rad_low, rad_hig


def get_BE_mass_1to4_ratio(percent_be_mass):
    """
    :param percent_be_mass: int. between 0 and 1. The ratio for how massive the cloud should be
    :return int. value between 1-4 representing the density of the Bonnor-Ebert Sphere with respect to the collapse
    density of 4 and the zero cloud mass of 0.
    """
    def eq(x):
        val = percent_be_mass - 1/(5 * sqrt(5)/16) * x * sqrt((x-1)/(3/5 * x**2))**3
        return val

    if 0 < percent_be_mass <= 1:
        mass_r = brenth(eq, 1, 4)

    elif percent_be_mass > 1:
        print("You've selected an unstable cloud that would collapse, Manually set ρ_real_over_ρ_pressure")
        mass_r = -1

    elif percent_be_mass < 0:
        print("Negative mass ratio selected")
        mass_r = -1

    else:
        print("Failed to calculate cloud mass, did you input a interger between 0 and 1? I got {} with type {}".format(percent_be_mass, type(percent_be_mass)))
        mass_r = -1

    return mass_r


def get_BE_mass_0to5sqrt5o16(ρ_normalised, override_percentage=-1):
    """
    Calculate the value between 0 and 5 sqrt(5)/16 for density
    Requires normalised density and returns the mass_ratio required. THe 5sqrt(5)/16 is NOT taken into account
    :param ρ_normalised: int. density of the cloud between 1 and 4 as calculated by get_BE_mass_1to4_ratio
    """
    # todo: far future, fix the printed states so it says 0 to 100% mBE
    if ρ_normalised < 1:
        print("Rho < 1 will result in zero mass specify override percentage")
        return override_percentage

    elif ρ_normalised > 4:
        print('Selected a mass over the maximum mass. Returning maximum value of 5 * sqrt(5)/16. Manually insert value')
        return 5 * sqrt(5)/16

    elif ρ_normalised == 4:
        print("Maximum stable mass selected")
        return 5 * sqrt(5)/16

    else:
        print("Solving for a mass cloud of " +
              str(ρ_normalised * sqrt((ρ_normalised - 1)/(3/5 * ρ_normalised**2))**3) +
              " m_BE, this value goes between 0 and approx 0.69877")
        return ρ_normalised * sqrt((ρ_normalised - 1)/(3/5 * ρ_normalised**2))**3


def axis_length_ratio_solver(start, stop):
    """
    Generate the length_2/length_1 axis ratio based on the length_3/length_1 axis ratio which is fed in.
    I.E generate the x/y axis ratio based on the z/x value
    :param start: int, the ratio value which we start solving at for a3/a1
    :param stop: int, the ratio value which we stop solving at for a3/a1
    :return: list of lists: the axis ratio lengths and Chandrasekhar values. Scaled as a_i/R^{(1/3)}
    """
    from scipy.optimize import brenth

    def equ_eq(length1, length2):
        """
        :param length1: int, the value that will be solved for a2/a1
        :param length2: int, the known value, a3/a1
        :return: int, the ratio length_2/length_1 consistent with length_3/length_1
        """
        Ax = get_Ax(x=1, y=length1, z=length2)
        Ay = get_Ay(x=1, y=length1, z=length2)
        Az = get_Az(x=1, y=length1, z=length2)
        return (Ax - Az * length2**2)/(Ay * length1**2 - Az * length2**2) - 3/length2**2 - 1

    vals = linspace(start, stop, 1000)  # This is the range of a3/a1 values we will feed to the functions
    sols = []  # This is a2/a1 values when using a given a3/a1

    for val in vals:
        # I found brenth to be the fastest solver.
        sols.append(brenth(equ_eq, val + 0.0001, 0.9999, args=(val,)))

    plot_vals_a1 = [1/(sols[i] * vals[i])**(1/3) for i in range(len(sols))]   # This gives a1/(a1 a2 a3)^3
    plot_vals_a2 = [1/(vals[i]/sols[i]**2)**(1/3) for i in range(len(sols))]  # This gives a2/(a1 a2 a3)^3
    plot_vals_a3 = [1/(sols[i]/vals[i]**2)**(1/3) for i in range(len(sols))]  # This gives a3/(a1 a2 a3)^3

    # The following values are for p=0 in Chandrasekhar's work.
    chan_a3a1 = [0.91355, 0.80902, 0.66913, 0.54464, 0.50000, 0.48481, 0.46947, 0.45399, 0.40674, 0.32557, 0.30902,
                 0.25882, 0.19081]
    # chan_a2a1 = [0.93188, 0.84112, 0.70687, 0.57787, 0.53013, 0.51373, 0.49714, 0.48040, 0.42898, 0.34052, 0.32254,
    #              0.26827, 0.19569]

    chan_a1 = [1.0551, 1.1369, 1.2835, 1.4701, 1.5567, 1.5894, 1.6242, 1.6613, 1.7896, 2.0816, 2.1568, 2.4330, 2.9919]
    chan_a2 = [0.9832, 0.9563, 0.9072, 0.8495, 0.8253, 0.8165, 0.8074, 0.7981, 0.7677, 0.7088, 0.6957, 0.6527, 0.5855]
    chan_a3 = [0.9639, 0.9198, 0.8588, 0.8007, 0.7784, 0.7706, 0.7625, 0.7542, 0.7279, 0.6777, 0.6665, 0.6297, 0.5709]

    return vals, sols, [plot_vals_a1, plot_vals_a2, plot_vals_a3], [chan_a3a1, chan_a1, chan_a2, chan_a3]


def internal_streaming_axis_length_ratio_solver(alpha):
    """
    Generates the a2/a1 and a3/a1 axis length ratios based on the input internal streaming rate
    Used to be called axis_length_ratio_solver_new from scratch2
    :param alpha: int: The internal streaming value
    :return: ints, a2/a1 and a3/a1
    """

    def a2oa1_eq(alpha):

        return sqrt(1 + 3/(alpha**2))

    def a3oa1_eq(alpha):

        return sqrt(2 * sqrt(alpha**2 + 3) - (alpha**2 + 3))

    return a2oa1_eq(alpha), a3oa1_eq(alpha)


def get_ai_lengths(alpha, ρ, ρ_tidal, return_only_a1=False):
    """
    Compute the dimenless lengths of the a1, a2, and a3 axis, Additionally finds the mass of the cloud
    rho_pressure is normalised out in these equations. The pressure is set to 1
    used to be called get_a3oa1_constant_line_new in scratch3
    :param alpha: int: the internal streaming value
    :param ρ: int: specified density between 1 and 4
    :param ρ_tidal: int: specified tidal strength
    :return: list, individual axis lengths and mass
    """

    def _calcuate_a1_length(ρ, ρ_tidal, a2oa1, a3oa1):
        """
        Calculate the length of the a1 axis using the brent hyperbolic method.
        used to be called get_a1_length_new from scratch3.
        :param ρ: int: density of the cloud, between 1 and 4
        :param ρ_tidal: int: specified tidal strength
        :return:
        """

        def _get_a1_length(a1):
            """
            Equation for root solver
            """
            return a1 ** 2 * a3oa1**2 * ρ_tidal - 5 * (1 - 1 / ρ)

        return brenth(_get_a1_length, 0.001, 300)

    # Obtain the a2/a1 and a3/a1 axis ratios for a given alpha.
    a2_over_a1, a3_over_a1 = internal_streaming_axis_length_ratio_solver(alpha)

    # Calculate the ai lengths
    a1_len = _calcuate_a1_length(ρ, ρ_tidal, a2_over_a1, a3_over_a1)
    a2_len = a2_over_a1 * a1_len
    a3_len = a3_over_a1 * a1_len
    # and mass
    mass = 4/3 * ρ * a1_len * a2_len * a3_len

    if return_only_a1:
        return a1_len
    else:
        return [a1_len, a2_len, a3_len], mass


def get_ai_lengths_chandrasekhar(a3_over_a1_index, specific_ρ=False):
    """
    Generate the ai lengths for the cloud based on the work by Chandrasekhar. Additionally calculates mass, mu, and
    tidal strength
    :param a3_over_a1_index. int. This is a tricky param. other methods need to be employed to find the correct index to
    give the correct a3/a1 length which is calculated by the 1000 index list generated by the function
    axis_length_ratio_solver - best to consult the author.
    :param specific_ρ: Choose a specific rho between 1 and 4 to use
    :return:
    """

    def _calculate_a1_length_chandrasekhar(ρ, ρ_tidal, a2_over_a1, a3_over_a1):
        """
        Calculate the length of the a1 axis as specified by Chandrasekhar
        :param ρ: int: density
        :param ρ_tidal: int: tidal strength
        :param a2_over_a1: int: a2 over a1 axis ratio
        :param a3_over_a1: int: a3 over a1 axis ratio
        :return: the a1 length as found by mullers method.
        """

        def _get_a1_length(a1):
            """
            Equation for the root solver
            """
            return a1 ** 2 * (3 * ρ_tidal / 1 - 9 / 2 * ρ / 1 * a2_over_a1 * a3_over_a1 * Ax) + 5 * (1 - 1 / ρ)

        Ax = get_Ax(x=1, y=a2_over_a1, z=a3_over_a1)

        return findroot(_get_a1_length, 0.5, solver="muller")  # Can instead do return brenth(_a1_length_equ, 0, 3)

    def _get_Omega_Chan(a2_over_a1, a3_over_a1):
        """
        Calculates the value for mu as specified in Chandrasekhar (year)
        :param a2oa1:
        :param a3oa1:
        :return:
        """
        return 2 * a2_over_a1 * a3_over_a1 * (get_Ax(x=1, y=a2_over_a1, z=a3_over_a1) - get_Az(x=1, y=a2_over_a1, z=a3_over_a1) * a3_over_a1 ** 2) / (
                    3 + a3_over_a1 ** 2)

    # generate the ratios and select the desired value
    a3_over_a1_list, a2_over_a1_list, _, _ = axis_length_ratio_solver(0.04, 0.99)
    a3_over_a1 = a3_over_a1_list[a3_over_a1_index]
    a2_over_a1 = a2_over_a1_list[a3_over_a1_index]

    if specific_ρ:
        ρs = [specific_ρ]
    else:
        ρs = arange(1.03, 3.8, 0.01)

    a1_lens = []
    a2_lens = []
    a3_lens = []
    mu_vals = []  # Will be given in units of pi G rho. Explicitly it is mu/(pi G rho)
    mass_vals = []
    ρ_tidal_vals = []

    for current_ρ in ρs:
        # Calculate mu and rho tides
        mu_Chan = _get_Omega_Chan(a2_over_a1, a3_over_a1)
        mu_vals.append(mu_Chan)
        ρ_tidal = (9/(4 * pi) * (pi * current_ρ) * mu_Chan)
        ρ_tidal_vals.append(ρ_tidal)

        # Calculate the axis lengths
        a1 = _calculate_a1_length_chandrasekhar(current_ρ, ρ_tidal, a2_over_a1, a3_over_a1)
        a1_lens.append(a1)
        a2_lens.append(a2_over_a1 * a1)
        a3_lens.append(a3_over_a1 * a1)

        # and the mass
        mass_vals.append(4/3 * current_ρ * a2_over_a1 * a3_over_a1 * a1**3)

    return [a1_lens, a2_lens, a3_lens], [mass_vals, ρ_tidal_vals, mu_vals]

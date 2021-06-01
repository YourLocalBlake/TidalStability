"""
Functions and classes that can be useful.
"""
from enum import IntEnum
from math import sqrt, pi
from mpmath import findroot, mp
from numpy import linspace, arange
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


class InitConIndex(IntEnum):
    """
    Enumerated number for array index for variables in the initial conditions object.
    """
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


class LengthIndex(IntEnum):
    """
    Enumerator number for array index for a_i lengths of the ellipsoid
    """
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
        print("and at this pressure equilibrium radii of r = {} and {} exist.".format(round(rad_low, 10),
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
              " m_BE, this value is highly non-linear and goes between 0 and approx 0.69877")
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
            return a1**2 * a3oa1**2 * ρ_tidal - 5 * (1 - 1 / ρ)

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


def get_ai_lengths_chandrasekhar(a3_over_a1_index, specific_ρ, a3_over_a1_list=False, a2_over_a1_list=False):
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
    if not a3_over_a1_list and not a2_over_a1_list:
        print("No a3/a1 and/or a2/a1 list provided - Calculating")
        a3_over_a1_list, a2_over_a1_list, _, _ = axis_length_ratio_solver(0.04, 0.99)
    a3_over_a1 = a3_over_a1_list[a3_over_a1_index]
    a2_over_a1 = a2_over_a1_list[a3_over_a1_index]

    # Calculate mu and rho tides
    mu_val = _get_Omega_Chan(a2_over_a1, a3_over_a1)  # in units of pi G rho. Explicitly it is mu/(pi G rho)
    ρ_tidal = (9/(4 * pi) * (pi * specific_ρ) * mu_val)

    # Calculate the axis lengths
    a1_len = _calculate_a1_length_chandrasekhar(specific_ρ, ρ_tidal, a2_over_a1, a3_over_a1)
    a2_len = a2_over_a1 * a1_len
    a3_len = a3_over_a1 * a1_len

    # and the mass
    mass_val = 4/3 * specific_ρ * a2_over_a1 * a3_over_a1 * a1_len**3

    return [float(a1_len), float(a2_len), float(a3_len)], float(mass_val), [specific_ρ, ρ_tidal, mu_val],


def get_physical_scale(solution, times, init_con, temp, pext, l_unit="cm"):
    """
    Calculate the physical scale of the cloud based on an input temperature and external pressure
    :param solution: Solution array
    :param times: list; list of times corresponding to the solver points
    :param init_con: InitialConditions object
    :param temp: int. Temperature in Kelvin
    :param pext: int: External pressure in Kelvin per cubic centimetre
    :param l_unit: string: the length unit to use, either pc (parsec), m (metre), or cm (centimetre) if defaulted.
    :return: SolutionPhysicalUnits: solution class object with physical parameters.
    """

    from tidal_stability.solve.utils import get_cs2
    from tidal_stability.physics_constants import GRAV, BOLTZMANN
    from tidal_stability.data_formats import SolutionPhysicalUnits

    pext = pext * BOLTZMANN  # Convert from Kelvin per cubic centimetre to pascals
    cs2 = get_cs2(temp=temp, mu=2.33)  # Returns in units of cm^2 s^-2

    len_scale = 3 * cs2 * 1/sqrt(4 * pi * GRAV * pext)       # Returns in units of cm
    time_scale = 3 * sqrt(cs2/(4 * pi * GRAV * pext))        # Returns in units of seconds
    mass_scale = 9 * cs2**2/(sqrt(4 * pi * GRAV**3 * pext))  # Returns in units of grams

    if l_unit.lower() == "pc":
        len_scale = len_scale * 3.240779289e-19
    if l_unit.lower() == "m":
        len_scale = len_scale * 0.01

    # give everything its corresponding time scale
    for index in (ODEIndex.xdot, ODEIndex.ydot, ODEIndex.zdot, ODEIndex.θdot, ODEIndex.ϕdot):
        solution[:, index] = solution[:, index] / time_scale
        times[:] = times[:] / time_scale
    # and length unit
    for index in (ODEIndex.x, ODEIndex.xdot, ODEIndex.y, ODEIndex.ydot, ODEIndex.z, ODEIndex.zdot):
        solution[:, index] = solution[:, index] * len_scale

    # todo: pull the densitiy and mass from init con and such.

    mass = init_con.mass_r * 1


    # # Calculate the respective densities
    # soln = SolutionPhysicalUnits(
    #     times=times,
    #     solution=solution,
    #     ρ_real=)




def calculate_solution_planes(x_val, y_val, z_val, θ, θdot, ϕdot, ρ_self, ρ_tides, style_heat=True, log_scale=True, constant_var="x"):
    """ calculate_solution_planes(0.64838746, 0.36108903, 0.34756538, 0, 0, -0.1/sqrt(ρ_pressure_over_ρ_tides), 1.1181313072462846, 1/4.4, "x")
    Calculate and plot the solution plane for a constant value and then ranging two other
    :return:
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from tidal_stability.solve.deriv_funcs import deriv_xdot_func, deriv_ydot_func, deriv_zdot_func
    from tidal_stability.solve.geometry import get_Ax, get_Ay, get_Az
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import cmasher as cmr
    cmap = cmr.iceburn
    nums = 100
    up_down_val = 0.05
    vmin_max_val = 0.05
    x_range = np.linspace(x_val - up_down_val, x_val + up_down_val, nums)
    y_range = np.linspace(y_val - up_down_val, y_val + up_down_val, nums)
    z_range = np.linspace(z_val - up_down_val, z_val + up_down_val, nums)

    if constant_var.lower() == "x":
        xddot_vals = []
        yddot_vals = []
        zddot_vals = []
        sumofsqs = []
        for y in y_range:
            for z in z_range:
                Ai_vals = [get_Ax(x=x_val, y=y, z=z), get_Ay(x=x_val, y=y, z=z), get_Az(x=x_val, y=y, z=z)]
                xddot_vals.append(deriv_xdot_func(
                    x=x_val, y=y, z=z, θ=θ, θdot=θdot, ϕdot=ϕdot, A1=Ai_vals[EllipIndex.x],
                    ρ_real_over_ρ_pressure=ρ_self, ρ_pressure_over_ρ_tides=1/ρ_tides)[0])
                yddot_vals.append(deriv_ydot_func(
                    x=x_val, y=y, z=z, θ=θ, θdot=θdot, ϕdot=ϕdot, A2=Ai_vals[EllipIndex.y],
                    ρ_real_over_ρ_pressure=ρ_self, ρ_pressure_over_ρ_tides=1/ρ_tides)[0])
                zddot_vals.append(deriv_zdot_func(
                    x=x_val, y=y, z=z, A3=Ai_vals[EllipIndex.z],
                    ρ_real_over_ρ_pressure=ρ_self, ρ_pressure_over_ρ_tides=1/ρ_tides)[0])
        for (x, y, z) in zip(xddot_vals, yddot_vals, zddot_vals):
            sumofsqs.append(x**2 + y**2 + z**2)

        plot_xddot = np.flipud(np.reshape(xddot_vals, (nums, nums)).T)
        plot_yddot = np.flipud(np.reshape(yddot_vals, (nums, nums)).T)
        plot_zddot = np.flipud(np.reshape(zddot_vals, (nums, nums)).T)
        plot_sumofsqs = np.flipud(np.reshape(sumofsqs, (nums, nums)).T)

        if log_scale:
            plot_xddot = np.log10(np.abs(plot_xddot))
            plot_yddot = np.log10(np.abs(plot_yddot))
            plot_zddot = np.log10(np.abs(plot_zddot))
            plot_sumofsqs = np.log10(plot_sumofsqs)

        if style_heat:

            min_plot = np.min(np.concatenate((plot_xddot, plot_yddot, plot_zddot)))
            max_plot = np.max(np.concatenate((plot_xddot, plot_yddot, plot_zddot)))

            fig_heatmaps, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True, sharex=True)
            f1 = ax1.imshow(plot_xddot, extent=(y_range[0], y_range[-1], z_range[0], z_range[-1]), vmin=min_plot, vmax=max_plot, cmap=cmap)
            f2 = ax2.imshow(plot_yddot, extent=(y_range[0], y_range[-1], z_range[0], z_range[-1]) , vmin=min_plot, vmax=max_plot, cmap=cmap)
            f3 = ax3.imshow(plot_zddot, extent=(y_range[0], y_range[-1], z_range[0], z_range[-1]) , vmin=min_plot, vmax=max_plot, cmap=cmap)
            f4 = ax4.imshow(plot_sumofsqs, extent=(y_range[0], y_range[-1], z_range[0], z_range[-1]) , vmin=np.min(plot_sumofsqs), vmax=np.max(plot_sumofsqs))
            fig_heatmaps.suptitle("Constant x = {}".format(x_val))
            ax1.set_xlabel("y value")
            ax2.set_xlabel("y value")
            ax3.set_xlabel("y value")
            ax4.set_xlabel("y value")
            ax1.set_ylabel("z value")
            ax4.set_ylabel("z value")
            ax1.tick_params(top=True, right=True, labelbottom=True, left=True, labelright=False, direction="inout")
            ax2.tick_params(top=True, right=True, labelbottom=True, left=True, labelright=False, direction="inout")
            ax3.tick_params(top=True, right=True, labelbottom=True, left=True, labelright=True,  direction="inout")
            ax4.tick_params(top=True, right=True, labelbottom=True, left=True, labelright=True, direction="inout")
            ax1.plot(y_val, z_val, color='white', marker="+", ms=12)
            ax2.plot(y_val, z_val, color='white', marker="+", ms=12)
            ax3.plot(y_val, z_val, color='white', marker="+", ms=12)
            ax4.plot(y_val, z_val, color='white', marker="+", ms=12)
            fig_heatmaps.delaxes(ax5)
            fig_heatmaps.delaxes(ax6)
            fig_heatmaps.subplots_adjust(wspace=0)
            fig_heatmaps.colorbar(f1, label="Log10 code value", aspect=5, cmap=cmap, ax=ax5)
            fig_heatmaps.colorbar(f4, label="Log10 code value", aspect=5, cmap=cmr.voltage, ax=ax6)
            ax1.set_title("xddot")
            ax2.set_title("yddot")
            ax3.set_title("zddot")
            ax4.set_title("sum in quad.")

        else:
        # Plot a 3d grid of the 3 planes

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            plot_y, plot_z = np.meshgrid(y_range, z_range)

            graph_xddot = ax.plot_surface(plot_y, plot_z, plot_xddot, cmap=cm.PuOr, linewidth=0, antialiased=False, vmin=-vmin_max_val, vmax=vmin_max_val)
            graph_yddot = ax.plot_surface(plot_y, plot_z, plot_yddot, cmap=cm.PiYG, linewidth=0, antialiased=False, vmin=-vmin_max_val, vmax=vmin_max_val)
            graph_zddot = ax.plot_surface(plot_y, plot_z, plot_zddot, cmap=cm.bwr,  linewidth=0, antialiased=False, vmin=-vmin_max_val, vmax=vmin_max_val)
            fig.colorbar(graph_xddot, shrink=0.5, aspect=5)
            fig.colorbar(graph_yddot, shrink=0.5, aspect=5)
            fig.colorbar(graph_zddot, shrink=0.5, aspect=5)

            plt.title("Constant x = {}".format(x_val))
            plt.xlabel("y value")
            plt.ylabel("z value")
            ax.set_zlabel("ddot value")



    elif constant_var.lower() == "y":
        xddot_vals = []
        yddot_vals = []
        zddot_vals = []
        sumofsqs = []

        for x in x_range:
            for z in z_range:
                Ai_vals = [get_Ax(x=x, y=y_val, z=z), get_Ay(x=x, y=y_val, z=z), get_Az(x=x, y=y_val, z=z)]
                xddot_vals.append(deriv_xdot_func(
                    x=x, y=y_val, z=z, θ=θ, θdot=θdot, ϕdot=ϕdot, A1=Ai_vals[EllipIndex.x],
                    ρ_real_over_ρ_pressure=ρ_self, ρ_pressure_over_ρ_tides=1/ρ_tides)[0])
                yddot_vals.append(deriv_ydot_func(
                    x=x, y=y_val, z=z, θ=θ, θdot=θdot, ϕdot=ϕdot, A2=Ai_vals[EllipIndex.y],
                    ρ_real_over_ρ_pressure=ρ_self, ρ_pressure_over_ρ_tides=1/ρ_tides)[0])
                zddot_vals.append(deriv_zdot_func(
                    x=x, y=y_val, z=z, A3=Ai_vals[EllipIndex.z],
                    ρ_real_over_ρ_pressure=ρ_self, ρ_pressure_over_ρ_tides=1/ρ_tides)[0])
        for (x, y, z) in zip(xddot_vals, yddot_vals, zddot_vals):
            sumofsqs.append(x**2 + y**2 + z**2)

        plot_xddot = np.flipud(np.reshape(xddot_vals, (nums, nums)))
        plot_yddot = np.flipud(np.reshape(yddot_vals, (nums, nums)))
        plot_zddot = np.flipud(np.reshape(zddot_vals, (nums, nums)))
        plot_sumofsqs = np.flipud(np.reshape(sumofsqs, (nums, nums)))

        if log_scale:
            plot_xddot = np.log10(np.abs(plot_xddot))
            plot_yddot = np.log10(np.abs(plot_yddot))
            plot_zddot = np.log10(np.abs(plot_zddot))
            plot_sumofsqs = np.log10(plot_sumofsqs)

        if style_heat:

            min_plot = np.min(np.concatenate((plot_xddot, plot_yddot, plot_zddot)))
            max_plot = np.max(np.concatenate((plot_xddot, plot_yddot, plot_zddot)))

            fig_heatmaps, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True, sharex=True)
            f1 = ax1.imshow(plot_xddot, extent=(x_range[0], x_range[-1], z_range[0], z_range[-1]), vmin=min_plot, vmax=max_plot, cmap=cmap)
            f2 = ax2.imshow(plot_yddot, extent=(x_range[0], x_range[-1], z_range[0], z_range[-1]) , vmin=min_plot, vmax=max_plot, cmap=cmap)
            f3 = ax3.imshow(plot_zddot, extent=(x_range[0], x_range[-1], z_range[0], z_range[-1]) , vmin=min_plot, vmax=max_plot, cmap=cmap)
            f4 = ax4.imshow(plot_sumofsqs, extent=(x_range[0], x_range[-1], z_range[0], z_range[-1]) , vmin=np.min(plot_sumofsqs), vmax=np.max(plot_sumofsqs))
            fig_heatmaps.suptitle("Constant y = {}".format(y_val))
            ax1.set_xlabel("x value")
            ax2.set_xlabel("x value")
            ax3.set_xlabel("x value")
            ax4.set_xlabel("x value")
            ax1.set_ylabel("z value")
            ax4.set_ylabel("sum in quad.")
            ax1.tick_params(top=True, right=True, labelbottom=True, left=True, labelright=False, direction="inout")
            ax2.tick_params(top=True, right=True, labelbottom=True, left=True, labelright=False, direction="inout")
            ax3.tick_params(top=True, right=True, labelbottom=True, left=True, labelright=True,  direction="inout")
            ax4.tick_params(top=True, right=True, labelbottom=True, left=True, labelright=True, direction="inout")
            ax1.plot(x_val, z_val, color='white', marker="+", ms=12)
            ax2.plot(x_val, z_val, color='white', marker="+", ms=12)
            ax3.plot(x_val, z_val, color='white', marker="+", ms=12)
            ax4.plot(x_val, z_val, color='white', marker="+", ms=12)
            fig_heatmaps.delaxes(ax5)
            fig_heatmaps.delaxes(ax6)
            fig_heatmaps.subplots_adjust(wspace=0)
            fig_heatmaps.colorbar(f1, label="Log10 code value", aspect=5, cmap=cmap, ax=ax5)
            fig_heatmaps.colorbar(f4, label="Log10 code value", aspect=5, cmap=cmr.voltage, ax=ax6)
            ax1.set_title("xddot")
            ax2.set_title("yddot")
            ax3.set_title("zddot")
            ax4.set_title("sum in quad.")

        else:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            plot_x, plot_z = np.meshgrid(x_range, z_range)

            graph_xddot = ax.plot_surface(plot_x, plot_z, plot_xddot, cmap=cm.PuOr, linewidth=0, antialiased=False,
                                          vmin=-vmin_max_val, vmax=vmin_max_val)
            graph_yddot = ax.plot_surface(plot_x, plot_z, plot_yddot, cmap=cm.PiYG, linewidth=0, antialiased=False,
                                          vmin=-vmin_max_val, vmax=vmin_max_val)
            graph_zddot = ax.plot_surface(plot_x, plot_z, plot_zddot, cmap=cm.bwr, linewidth=0, antialiased=False,
                                          vmin=-vmin_max_val, vmax=vmin_max_val)
            fig.colorbar(graph_xddot, shrink=0.5, aspect=5)
            fig.colorbar(graph_yddot, shrink=0.5, aspect=5)
            fig.colorbar(graph_zddot, shrink=0.5, aspect=5)

            plt.title("Constant y = {}".format(y_val))
            plt.xlabel("x value")
            plt.ylabel("z value")
            ax.set_zlabel("ddot value")

    elif constant_var.lower() == "z":
        xddot_vals = []
        yddot_vals = []
        zddot_vals = []
        sumofsqs = []
        for x in x_range:
            for y in y_range:
                Ai_vals = [get_Ax(x=x, y=y, z=z_val), get_Ay(x=x, y=y, z=z_val), get_Az(x=x, y=y, z=z_val)]
                xddot_vals.append(deriv_xdot_func(
                    x=x, y=y, z=z_val, θ=θ, θdot=θdot, ϕdot=ϕdot, A1=Ai_vals[EllipIndex.x],
                    ρ_real_over_ρ_pressure=ρ_self, ρ_pressure_over_ρ_tides=1/ρ_tides)[0])
                yddot_vals.append(deriv_ydot_func(
                    x=x, y=y, z=z_val, θ=θ, θdot=θdot, ϕdot=ϕdot, A2=Ai_vals[EllipIndex.y],
                    ρ_real_over_ρ_pressure=ρ_self, ρ_pressure_over_ρ_tides=1/ρ_tides)[0])
                zddot_vals.append(deriv_zdot_func(
                    x=x, y=y, z=z_val, A3=Ai_vals[EllipIndex.z],
                    ρ_real_over_ρ_pressure=ρ_self, ρ_pressure_over_ρ_tides=1/ρ_tides)[0])
        for (x, y, z) in zip(xddot_vals, yddot_vals, zddot_vals):
            sumofsqs.append(x**2 + y**2 + z**2)

        plot_xddot = np.flipud(np.reshape(xddot_vals, (nums, nums)))
        plot_yddot = np.flipud(np.reshape(yddot_vals, (nums, nums)))
        plot_zddot = np.flipud(np.reshape(zddot_vals, (nums, nums)))
        plot_sumofsqs = np.flipud(np.reshape(sumofsqs, (nums, nums)))

        if log_scale:
            plot_xddot = np.log10(np.abs(plot_xddot))
            plot_yddot = np.log10(np.abs(plot_yddot))
            plot_zddot = np.log10(np.abs(plot_zddot))
            plot_sumofsqs = np.log10(plot_sumofsqs)

        if style_heat:

            min_plot = np.min(np.concatenate((plot_xddot, plot_yddot, plot_zddot)))
            max_plot = np.max(np.concatenate((plot_xddot, plot_yddot, plot_zddot)))

            fig_heatmaps, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True, sharex=True)
            f1 = ax1.imshow(plot_xddot, extent=(x_range[0], x_range[-1], y_range[0], y_range[-1]), vmin=min_plot, vmax=max_plot, cmap=cmap)
            f2 = ax2.imshow(plot_yddot, extent=(x_range[0], x_range[-1], y_range[0], y_range[-1]) , vmin=min_plot, vmax=max_plot, cmap=cmap)
            f3 = ax3.imshow(plot_zddot, extent=(x_range[0], x_range[-1], y_range[0], y_range[-1]) , vmin=min_plot, vmax=max_plot, cmap=cmap)
            f4 = ax4.imshow(plot_sumofsqs, extent=(x_range[0], x_range[-1], y_range[0], y_range[-1]) , vmin=np.min(plot_sumofsqs), vmax=np.max(plot_sumofsqs))
            fig_heatmaps.suptitle("Constant z = {}".format(z_val))
            ax1.set_xlabel("x value")
            ax2.set_xlabel("x value")
            ax3.set_xlabel("x value")
            ax4.set_xlabel("x value")
            ax1.set_ylabel("y value")
            ax4.set_ylabel("sum in quad.")
            ax1.tick_params(top=True, right=True, labelbottom=True, left=True, labelright=False, direction="inout")
            ax2.tick_params(top=True, right=True, labelbottom=True, left=True, labelright=False, direction="inout")
            ax3.tick_params(top=True, right=True, labelbottom=True, left=True, labelright=True,  direction="inout")
            ax4.tick_params(top=True, right=True, labelbottom=True, left=True, labelright=True, direction="inout")
            ax1.plot(x_val, y_val, color='white', marker="+", ms=12)
            ax2.plot(x_val, y_val, color='white', marker="+", ms=12)
            ax3.plot(x_val, y_val, color='white', marker="+", ms=12)
            ax4.plot(x_val, y_val, color='white', marker="+", ms=12)
            fig_heatmaps.delaxes(ax5)
            fig_heatmaps.delaxes(ax6)
            fig_heatmaps.subplots_adjust(wspace=0)
            fig_heatmaps.colorbar(f1, label="Log10 code value", aspect=5, cmap=cmap, ax=ax5)
            fig_heatmaps.colorbar(f4, label="Log10 code value", aspect=5, cmap=cmr.voltage, ax=ax6)
            ax1.set_title("xddot")
            ax2.set_title("yddot")
            ax3.set_title("zddot")
            ax4.set_title("sum in quad.")

        else:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            plot_x, plot_y = np.meshgrid(x_range, y_range)

            graph_xddot = ax.plot_surface(plot_x, plot_y, plot_xddot, cmap=cm.PuOr, linewidth=0, antialiased=False,
                                          vmin=-vmin_max_val, vmax=vmin_max_val)
            graph_yddot = ax.plot_surface(plot_x, plot_y, plot_yddot, cmap=cm.PiYG, linewidth=0, antialiased=False,
                                          vmin=-vmin_max_val, vmax=vmin_max_val)
            graph_zddot = ax.plot_surface(plot_x, plot_y, plot_zddot, cmap=cm.bwr, linewidth=0, antialiased=False,
                                          vmin=-vmin_max_val, vmax=vmin_max_val)
            fig.colorbar(graph_xddot, shrink=0.5, aspect=5)
            fig.colorbar(graph_yddot, shrink=0.5, aspect=5)
            fig.colorbar(graph_zddot, shrink=0.5, aspect=5)

            plt.title("Constant z = {}".format(z_val))
            plt.xlabel("x value")
            plt.ylabel("y value")
            ax.set_zlabel("ddot value")


    else:
        raise SystemExit("Incorrect variable selected, got {}, expected x, y, or z".format(constant_var))

    # fig, ax = plt.subplot(1, 3)
    # calculate_solution_planes(0.5, 0.5, 0.5, 0, 0, -0.1, 1.18, 0.01, "x")


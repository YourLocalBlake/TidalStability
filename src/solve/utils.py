"""
Useful utilises for the solver.
"""

from numpy import linspace, sign

from src.physics_constants import GRAV, BOLTZMANN, H1MASS
from src.utils import ODEIndex


def get_Ω2(*, dis, mass_bh):
    """
    Compute the angular velocity squared of the cloud around the blackhole.
    """
    # TODO: Move this and cs2 to other utils folder.
    return GRAV * mass_bh/dis**3


def get_cs2(*, temp, mu):
    """
    Compute the sound speed squared
    """
    return BOLTZMANN * temp / (mu * H1MASS)


def xy_close_stop_generator(target_diff):
    """
    return function which stops where x and y are 'target_diff' amount close to one another
    """

    def rootfunc(time, params, out):
        """
        The function required by sundials
        """
        out[0] = target_diff - abs(params[ODEIndex.x] - params[ODEIndex.y])
        return 0

    return rootfunc


def get_jump_amount(x, y, xdot, ydot, override=False):
    """
    calculate the time required for the jump to swap the values of x and y
    """
    if override:
        return override

    diff = x - y
    diff_sign = sign(diff)

    time_attempts = linspace(1e-6, 1e-1, 1000000)

    # Only calculating the difference to linear order
    for dt in time_attempts:
        jumped_x = x + xdot * dt
        jumped_y = y + ydot * dt
        if diff_sign == 1:
            if jumped_x - jumped_y < - 1.001 * diff:
                print("Jumping over a time of {}".format(dt))
                return dt
        if diff_sign == -1:
            if jumped_x - jumped_y > - 1.001 * diff:
                print("Jumping over a time of {}".format(dt))
                return dt

    print("No jump time was sufficient to cross, check values or apply override - check solution convergence")
    raise SystemExit


def ontstop_cont():
    """
    Return function which stops the ODE solver when called
    """
    def tstop_func(time, params, out):
        """
        The function required by sundials
        """
        return 0

    return tstop_func


def ontstop_stop():
    """
    Return function which stops the ODE solver when called
    """

    def tstop_func(time, params, out):
        """
        The function required by sundials
        """
        return 1

    return tstop_func

"""
Derivatives for the equations of motion
The returned zeros as the second variable of each ODE equation represents a object which can be changed to return any
value wanted for analysis. By returning the full equation in the 2nd spot the individual time steps solved by the
numerical integrator will be returned
"""

from math import sin, cos, sqrt


def deriv_x_func(*, xdot):
    """
    Compute the derivative of the x axis of the ellipsoid
    """
    return xdot, xdot


def deriv_xdot_func(*, x, y, z, θ, θdot, ϕdot, A1, ρ_real_over_ρ_pressure, ρ_pressure_over_ρ_tides):
    """
    Compute the derivative of the derivative of the axis of the ellipsoid
    """
    # x * y * z * ρ_real_over_ρ_pressure  is chosen instead of mass_r as it accounts for the scale factor normalisation
    # that arises from the BE mass ratio deviation from sphericity.
    # They are physically interchangable
    return (
        1/x * (
            + ϕdot * (+ ϕdot * x**2 + 2 * x * y * (+ θdot + 1/sqrt(ρ_pressure_over_ρ_tides)))
            + x**2 * (
                + θdot * (θdot + 2/sqrt(ρ_pressure_over_ρ_tides))
                + 3/ρ_pressure_over_ρ_tides * cos(θ)**2
                - 9/2 * x * y * z * ρ_real_over_ρ_pressure * A1
            )
            + 5 * (1 - 1/ρ_real_over_ρ_pressure)
        )
        , 0
    )


def deriv_y_func(*, ydot):
    """
    Compute the derivative of the y axis of the ellipsoid
    """
    return ydot, ydot


def deriv_ydot_func(*, x, y, z, θ, θdot, ϕdot, A2, ρ_real_over_ρ_pressure, ρ_pressure_over_ρ_tides):
    """
    Compute the derivative of the derivative of the axis of the ellipsoid ellipsoid
    """
    return (
        1/y * (
            + ϕdot * (+ ϕdot * y**2 + 2 * x * y * (+ θdot + 1/sqrt(ρ_pressure_over_ρ_tides)))
            + y**2 * (
                + θdot * (θdot + 2/sqrt(ρ_pressure_over_ρ_tides))
                + 3/ρ_pressure_over_ρ_tides * sin(θ)**2
                - 9/2 * A2 * x * y * z * ρ_real_over_ρ_pressure
            )
            + 5 * (1 - 1/ρ_real_over_ρ_pressure)
        )
        , 0
    )


def deriv_z_func(*, zdot):
    """
    Compute the derivative of the z axis of the ellipsoid
    """
    return zdot, zdot


def deriv_zdot_func(*, x, y, z, A3, ρ_real_over_ρ_pressure, ρ_pressure_over_ρ_tides):
    """
    Compute the derivative of the derivative of the axis of the ellipsoid
    """
    return (
        1/z * (
            - z**2 * (
                + 9/2 * A3 * x * y * z * ρ_real_over_ρ_pressure
                + 1/ρ_pressure_over_ρ_tides
            )
            + 5 * (1 - 1/ρ_real_over_ρ_pressure)
        )
        , 0
    )


def deriv_θ_func(*, θdot):
    """
    Compute the derivative of the θ axis of the ellipsoid
    """
    return θdot, θdot


def deriv_θdot_func(*, x, xdot, y, ydot, θ, θdot, ϕdot, ρ_pressure_over_ρ_tides):
    """
    Compute the derivative of the derivative of the θ axis of the ellipsoid
    """
    return (
        1/((-x + y) * (x + y)) * (
                + 2 * (xdot * x - ydot * y) * (θdot + sqrt(1/ρ_pressure_over_ρ_tides))
                + 2 * ϕdot * (- xdot * y + ydot * x)
                + 3/2 * sin(2 * θ) * (x**2 + y**2) * 1/ρ_pressure_over_ρ_tides
        )
        , 0
    )


def deriv_ϕ_func(*, ϕdot):
    """
    Compute the derivative of the ϕ axis of the ellipsoid
    """
    return ϕdot, ϕdot


def deriv_ϕθdot_func(*,  x, xdot, y, ydot, θ, θdot, ϕdot, ρ_pressure_over_ρ_tides):
    """
    Compute the derivative of the derivative of the ϕ axis of the ellipsoid
    """
    return (
        1/(x + y) * (
            - 2 * (xdot + ydot) * (θdot + ϕdot + sqrt(1/ρ_pressure_over_ρ_tides))
            - 3/2 * sin(2 * θ) * (x - y) * 1/ρ_pressure_over_ρ_tides
        )
        , 0
    )


def deriv_ϕdot_func(*,  x, xdot, y, ydot, θ, θdot, ϕdot, ρ_tides_over_ρ_pressure, sqrt_ρ_tides_over_ρ_pressure):
    """
    Compute the derivative of the derivative of the ϕ axis of the ellipsoid
    """

    return (
        1/((x - y) * (x + y)) * (
            + 2 * ϕdot * (- xdot * x + ydot * y)
            + 2 * (xdot * y - ydot * x) * (θdot + sqrt_ρ_tides_over_ρ_pressure)
            + 3 * ρ_tides_over_ρ_pressure * x * y * sin(2 * θ)
        )
        , 0
    )

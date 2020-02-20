import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi
from time import time as current_time

from src.data_formats import InitialConditions, ConfigParams
from src.utils import (ODEIndex, get_BE_mass_0to5sqrt5o16, get_BE_mass_1to4_ratio,
                       LengthIndex)
from src.internal_streaming_equilibria import get_rot_equ_axis_lengths
from src.solve.solution import solution


# System params
# Set system values
ratio = 0.1                                     # Specify a ratio for the Bonnor-Ebert mass. e.g. 10% = 0.1
# Calculate required values
ρ_real_over_ρ_pressure = get_BE_mass_1to4_ratio(ratio)

# Set system values, NOTE: init_ϕ_v should be set as "alpha/sqrt(ρ_pressure_over_ρ_tides)"
alpha = -0.1  # Internal streaming rate
ρ_pressure_over_ρ_tides = 4.4  # Choose tidal strength
# Calculate required values
print("In, r/p = {}, p/t={}".format(ρ_real_over_ρ_pressure, ρ_pressure_over_ρ_tides))
ai_lens, mass, ρs = get_rot_equ_axis_lengths(alpha, ρ_real_over_ρ_pressure, 1 / ρ_pressure_over_ρ_tides, final_only=True)

from src.utils import calculate_solution_planes
calculate_solution_planes(ai_lens[0], ai_lens[1], ai_lens[2], 0, 0,
                          alpha/sqrt(ρ_pressure_over_ρ_tides),
                          ρs[0], # ρ_real_over_ρ_pressure,
                          1/ρ_pressure_over_ρ_tides,
                          style_heat=True,
                          log_scale=True,
                          constant_var="x")
calculate_solution_planes(ai_lens[0], ai_lens[1], ai_lens[2], 0, 0,
                          alpha/sqrt(ρ_pressure_over_ρ_tides),
                          ρs[0],
                          1/ρ_pressure_over_ρ_tides,
                          style_heat=True,
                          log_scale=True,
                          constant_var="y")
calculate_solution_planes(ai_lens[0], ai_lens[1], ai_lens[2], 0, 0,
                          alpha/sqrt(ρ_pressure_over_ρ_tides),
                          ρs[0],
                          1/ρ_pressure_over_ρ_tides,
                          style_heat=True,
                          log_scale=True,
                          constant_var="z")
### plotting ai lens
a1_lens = []
a2_lens = []
a3_lens = []

for j, i, in enumerate(ai_lens):
    if ρs[-1][j]:
        a1_lens.append(i[0])
        a2_lens.append(i[1])
        a3_lens.append(i[2])
lens = range(len(a1_lens))
plt.scatter(lens, a1_lens, label="a1")
plt.scatter(lens, a2_lens, label="a2")
plt.scatter(lens, a3_lens, label="a3")
plt.legend(loc='best')
plt.show()

#
# ρ_real_over_ρ_pressure = ρs[0] - 1 / 1000
# ρ_real_over_ρ_tides = ρ_real_over_ρ_pressure * ρ_pressure_over_ρ_tides
# mass_r = get_BE_mass_0to5sqrt5o16(ρ_real_over_ρ_pressure)
# print("Out, r/p = {}, p/t={}".format(ρ_real_over_ρ_pressure, ρ_pressure_over_ρ_tides))
# equ_radius = 1
#
# init_a1 = ai_lens[LengthIndex.a1]               # Initial a1 axis length of ellipsoid, a1 is usually scaled to 1.005
# init_a2 = ai_lens[LengthIndex.a2]               # Initial a2 axis length, a2 is usually scaled to 0.995
# init_a3 = ai_lens[LengthIndex.a3]               # Initial a3 axis length, a3 is usually scaled to 1.000
# init_θ = 0                                      # Initial θ angle in radians
# init_ϕ = 0                                      # Initial ϕ angle in radians
# init_a1_v = 0                                   # Initial velocity of a1 axis, scaled by equ. radius
# init_a2_v = 0                                   # Initial velocity of a2 axis, scaled by equ. radius
# init_a3_v = 0                                   # Initial velocity of a3 axis, scaled by equ. radius
# init_θ_v = 0                                    # Initial velocity of the θ rotation angle
# init_ϕ_v = alpha/sqrt(ρ_pressure_over_ρ_tides)  # Initial velocity of the ϕ rotation angle
#
#
# start_time = 0                                  # Time to start the simulation
# stop_time = 25                                  # Time to stop simulation in seconds - will be scaled by time unit
# time_steps = 10e4                               # Number of time steps to solve during the simulation
# max_steps = 10000                               # Max number of iterations the solver may take to solve a timestep
# relative_tolerance = 1e-12                      # Relative tolerance of the solver
# absolute_tolerance = 1e-12                      # Absolute tolerance of the solver
# enable_taylor_jump = False                      # Choice to jump axis length changes with a taylor series approximation
# taylor_jumps_num = 1                            # how many taylor jumps the system is allowed.
# taylor_jump_threshold = 0.005                   # Distance between a1 and a2 axis at which to employ taylor series
# enable_tstop = False                            # Choice to stop the simulation at certain times
# orbital_period = 2 * pi * sqrt(ρ_pressure_over_ρ_tides)
# tstop_times = [orbital_period, orbital_period + orbital_period/100]
# tstop_changes = [{}]                            # When the tstop occurs, choose how you want to alter the system
#                                                 # These values represent multipliers based on original value each
#                                                 # list item is a dictionary
#
# save_data = False                                # Would you like to create a hdf5 file with the results?
# folder_name = "solved_odes"                      # Save folder name
#
# initial_conditions = InitialConditions(
#     ode_init_con=[init_a1 * equ_radius, init_a1_v * equ_radius,
#                   init_a2 * equ_radius, init_a2_v * equ_radius,
#                   init_a3 * equ_radius, init_a3_v * equ_radius,
#                   init_θ, init_θ_v,
#                   init_ϕ, init_ϕ_v],
#     ρ_real_over_ρ_tides=ρ_real_over_ρ_tides,
#     ρ_pressure_over_ρ_tides=ρ_pressure_over_ρ_tides,
#     ρ_real_over_ρ_pressure=ρ_real_over_ρ_pressure,
#     mass_r=mass_r,
#     equ_radius=equ_radius,
#     after_tstop_params=tstop_changes
# )
# solver_config = ConfigParams(
#     start=start_time,
#     stop=stop_time,
#     num_time=time_steps,
#     max_steps=max_steps,
#     relative_tolerance=relative_tolerance,
#     absolute_tolerance=absolute_tolerance,
#     enable_taylor_jump=enable_taylor_jump,
#     taylor_jumps_num=taylor_jumps_num,
#     taylor_jump_threshold=taylor_jump_threshold,
#     enable_tstop=enable_tstop,
#     tstop_times=tstop_times
# )
# # Compute the solution
# sol_start_time = current_time()
# soln, time, flag, internal_data = solution(initial_conditions=initial_conditions, solver_config=solver_config,
#                                            save_data=save_data, folder_name=folder_name)
# print("Time taken to compute calculation: " + str(current_time() - sol_start_time) + " seconds")
#
#
# print("Density ratios were:")
# print("gravity to tides: ", ρ_real_over_ρ_tides)
# print("pressure to tides: ", ρ_pressure_over_ρ_tides)
# print("density to pressure ratio: ", ρ_real_over_ρ_pressure)
#
# plt.subplot(3, 2, 1)
# plt.plot(time, soln[:, ODEIndex.x], label="a1", color='black')
# plt.plot(time, soln[:, ODEIndex.y], label="a2")
# plt.plot(time, soln[:, ODEIndex.z], label="a3", color='red')
# plt.axvline(x=2 * pi * 1 * 1 * sqrt(ρ_pressure_over_ρ_tides), color='grey', alpha=0.4, ls="--", label="Orbital time")
# for i in np.arange(2, 100, 1):
#     if 2 * pi * i * sqrt(ρ_pressure_over_ρ_tides) < time[-1]:
#         plt.axvline(x=2 * pi * i * 1 * sqrt(ρ_pressure_over_ρ_tides), color='grey', alpha=0.4, ls="--")
# if enable_tstop:
#     plt.axvspan(tstop_times[0], tstop_times[1], color='grey', alpha=0.4)
# plt.legend(loc='upper left')
# plt.xlabel("Time (T)")
# plt.ylabel("Length (L)")
#
# plt.subplot(3, 2, 2)
# plt.plot(time, soln[:, ODEIndex.xdot], label="d/dt a1", color='black')
# plt.plot(time, soln[:, ODEIndex.ydot], label="d/dt a2")
# plt.plot(time, soln[:, ODEIndex.zdot], label="d/dt a3", color='red')
# if enable_tstop:
#     plt.axvspan(tstop_times[0], tstop_times[1], color='grey', alpha=0.4)
# plt.legend(loc='best')
# plt.xlabel("Time (T)")
# plt.ylabel("Velocity (cs)")
#
# plt.subplot(3, 2, 3)
# plt.plot(time,
#          (1/(soln[:, ODEIndex.x] * soln[:, ODEIndex.y] * soln[:, ODEIndex.z]) *
#           (soln[0, ODEIndex.x] * soln[0, ODEIndex.y] * soln[0, ODEIndex.z])),
#          label="density")
# if enable_tstop:
#     plt.axvspan(tstop_times[0], tstop_times[1], color='grey', alpha=0.4)
# plt.legend(loc='best')
# plt.xlabel("Time (T)")
# plt.ylabel("Density (scaled to initial density)")
#
# plt.subplot(3, 2, 4)
# from math import degrees
# plt.plot(time, [degrees(i) for i in soln[:, ODEIndex.θ]], label="θ")
# plt.plot(time, [degrees(i-j) for (i, j) in zip(soln[:, ODEIndex.ϕ], soln[:, ODEIndex.θ])], label="ϕ")
# plt.plot(time, [degrees(i) for i in soln[:, ODEIndex.ϕ]], label="ϕ + θ code variable")
# plt.axvline(x=2 * pi * 1 * 1 * sqrt(ρ_pressure_over_ρ_tides), color='grey', alpha=0.4, ls="--", label="Orbital time")
# for i in np.arange(2, 100, 1):
#     if 2 * pi * i * sqrt(ρ_pressure_over_ρ_tides) < time[-1]:
#         plt.axvline(x=2 * pi* i * 1 * sqrt(ρ_pressure_over_ρ_tides), color='grey', alpha=0.4, ls="--")
# if enable_tstop:
#     plt.axvspan(tstop_times[0], tstop_times[1], color='grey', alpha=0.4)
# plt.legend(loc='best')
# plt.xlabel("Time (T)")
# plt.ylabel("Theta (degrees)")
#
# plt.subplot(3, 2, 5)
# plt.plot(time, soln[:, ODEIndex.θdot], label="θdot")
# plt.plot(time, (soln[:, ODEIndex.ϕdot] - soln[:, ODEIndex.θdot]), label="ϕdot")
# if enable_tstop:
#     plt.axvspan(tstop_times[0], tstop_times[1], color='grey', alpha=0.4)
# plt.legend(loc='best')
# plt.xlabel("Time (T)")
# plt.ylabel("d/dt theta (1/T)")
#
# plt.subplot(3, 2, 6)
# init_den = 3 * mass_r/(4 * pi * soln[0, ODEIndex.x] * soln[0, ODEIndex.y] * soln[0, ODEIndex.z])
# den = 3 * mass_r/(4 * pi * soln[:, ODEIndex.x] * soln[:, ODEIndex.y] * soln[:, ODEIndex.z])
# plt.axhline(y=4/ρ_real_over_ρ_pressure, linestyle="--", label="4 P/cs2 (1/init den)", color='black')
# plt.axhline(y=1/ρ_real_over_ρ_pressure, linestyle="--", label="P/cs2 (1/init den)", color='blue')
# plt.plot(time, den/init_den, label="Real density", color="red")
# plt.axhspan(0, 4 * 1/ρ_real_over_ρ_tides* 1/(9 * 0.090068) , color='grey', alpha=0.4, label="Roche Unstable")
# if enable_tstop:
#     plt.axvspan(tstop_times[0], tstop_times[1], color='black', alpha=0.8)
# plt.ylim(0)
# plt.legend(loc='best')
# plt.xlabel("Time (T)")
# plt.ylabel("Rho (init Rho)")
# plt.show()



###############################
# old equilibria equations.
from src.solve.deriv_funcs import deriv_xdot_func, deriv_ydot_func, deriv_zdot_func
# def deriv_xdot_func_vec_new_new(*, vec_input, θ, θdot, ϕdot, A1, ρ_real_over_ρ_pressure, ρ_pressure_over_ρ_tides):
#     """
#     Compute the derivative of the derivative of the axis of the ellipsoid
#     """
#     x = vec_input[0]
#     y = vec_input[1]
#     z = vec_input[2]
#     return deriv_xdot_func(x=x, y=y, z=z, θ=θ, θdot=θdot, ϕdot=ϕdot, A1=A1, ρ_real_over_ρ_pressure=ρ_real_over_ρ_pressure, ρ_pressure_over_ρ_tides=ρ_pressure_over_ρ_tides)


def deriv_xdot_func_vec_new(vec_input, θ, θdot, ϕdot, A1, ρ_real_over_ρ_pressure, ρ_pressure_over_ρ_tides):
    """
    Compute the derivative of the derivative of the axis of the ellipsoid
    """
    x = vec_input[0]
    y = vec_input[1]
    z = vec_input[2]
    return 1/x * (
            + ϕdot * (+ ϕdot * x**2 + 2 * x * y * (+ θdot + 1/sqrt(ρ_pressure_over_ρ_tides)))
            + x**2 * (
                + θdot * (θdot + 2/sqrt(ρ_pressure_over_ρ_tides))
                + 3/ρ_pressure_over_ρ_tides * cos(θ)**2
                - 9/2 * x * y * z * ρ_real_over_ρ_pressure * A1
            )
            + 5 * (1 - 1/ρ_real_over_ρ_pressure)
        )




def deriv_ydot_func_vec_new(vec_input, θ, θdot, ϕdot, A2, ρ_real_over_ρ_pressure, ρ_pressure_over_ρ_tides):
    """
    Compute the derivative of the derivative of the axis of the ellipsoid ellipsoid
    """
    x = vec_input[0]
    y = vec_input[1]
    z = vec_input[2]
    return 1/y * (
            + ϕdot * (+ ϕdot * y**2 + 2 * x * y * (+ θdot + 1/sqrt(ρ_pressure_over_ρ_tides)))
            + y**2 * (
                + θdot * (θdot + 2/sqrt(ρ_pressure_over_ρ_tides))
                + 3/ρ_pressure_over_ρ_tides * sin(θ)**2
                - 9/2 * A2 * x * y * z * ρ_real_over_ρ_pressure
            )
            + 5 * (1 - 1/ρ_real_over_ρ_pressure)
        )

def deriv_zdot_func_vec_new(vec_input, A3, ρ_real_over_ρ_pressure, ρ_pressure_over_ρ_tides):
    """
    Compute the derivative of the derivative of the axis of the ellipsoid
    """
    x = vec_input[0]
    y = vec_input[1]
    z = vec_input[2]
    return 1/z * (
            - z**2 * (
                + 9/2 * A3 * x * y * z * ρ_real_over_ρ_pressure
                + 1/ρ_pressure_over_ρ_tides
            )
            + 5 * (1 - 1/ρ_real_over_ρ_pressure)
        )
#
# print("suc?", a1_new.success)
#         print("mes", a1_new.message)
#         print("vals are ", )
#         Ai = [
#             get_Ax(x=a1_new.x[EllipIndex.x], y=a1_new.x[EllipIndex.y], z=a1_new.x[EllipIndex.z]),
#             get_Ay(x=a1_new.x[EllipIndex.x], y=a1_new.x[EllipIndex.y], z=a1_new.x[EllipIndex.z]),
#             get_Az(x=a1_new.x[EllipIndex.x], y=a1_new.x[EllipIndex.y], z=a1_new.x[EllipIndex.z])]
#         print("x", deriv_xdot_func_vec(vec_input=a1_new.x, θ=0, θdot=0, ϕdot=ϕdot, A1=Ai[0], ρ_real_over_ρ_pressure=ρ_init, ρ_pressure_over_ρ_tides=1/ρ_tides))
#         print("y", deriv_ydot_func_vec(vec_input=a1_new.x, θ=0, θdot=0, ϕdot=ϕdot, A2=Ai[1], ρ_real_over_ρ_pressure=ρ_init, ρ_pressure_over_ρ_tides=1/ρ_tides))
#         print("z", deriv_zdot_func_vec(vec_input=a1_new.x, A3=Ai[2], ρ_real_over_ρ_pressure=ρ_init, ρ_pressure_over_ρ_tides=1/ρ_tides))
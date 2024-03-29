from math import sqrt, pi
from time import time as current_time
import numpy as np
import matplotlib.pyplot as plt

from tidal_stability.utils import ODEIndex, get_BE_equilibrium_radius, get_BE_mass_0to5sqrt5o16, get_BE_mass_1to4_ratio
from tidal_stability.data_formats import InitialConditions, ConfigParams
from tidal_stability.solve.solution import solution

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                                                SET SYSTEM PARAMETERS                                                 #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################
# System params
# Specify a ratio for the Bonnor-Ebert mass. e.g. 10% = 0.1
ratio = 0.9
ρ_real_over_ρ_tides = 25

ρ_real_over_ρ_pressure = get_BE_mass_1to4_ratio(ratio)
ρ_pressure_over_ρ_tides = ρ_real_over_ρ_tides/ρ_real_over_ρ_pressure
mass_r = get_BE_mass_0to5sqrt5o16(ρ_real_over_ρ_pressure)

# Equation initial conditions
init_a1 = 1.005                                 # Initial a1 axis length of ellipsoid, a1 is usually scaled to 1
init_a2 = 0.995                                 # Initial a2 axis length
init_a3 = 1.000                                 # Initial a3 axis length
init_θ = 0                                      # Initial θ angle in radians
init_ϕ = 0                                      # Initial ϕ angle in radians
init_a1_v = 0                                   # Initial velocity of a1 axis, scaled by equ. radius
init_a2_v = 0                                   # Initial velocity of a2 axis, scaled by equ. radius
init_a3_v = 0                                   # Initial velocity of a3 axis, scaled by equ. radius
init_θ_v = 0                                    # Initial velocity of the θ rotation angle
init_ϕ_v = 0                                    # Initial velocity of the ϕ rotation angle

start_time = 0                                  # Time to start the simulation
stop_time = 25                                  # Time to stop simulation in seconds - will be scaled by time unit
time_steps = 10e4                               # Number of time steps to solve during the simulation
max_steps = 10000                               # Max number of iterations the solver may take to solve a timestep
relative_tolerance = 1e-12                      # Relative tolerance of the solver
absolute_tolerance = 1e-12                      # Absolute tolerance of the solver
enable_taylor_jump = False                      # Choice to jump axis length changes with a taylor series approximation
taylor_jumps_num = 1                            # how many taylor jumps the system is allowed.
taylor_jump_threshold = 0.005                   # Distance between a1 and a2 axis at which to employ taylor series
enable_tstop = False                            # Choice to stop the simulation at certain times
orbital_period = 2 * pi * sqrt(ρ_pressure_over_ρ_tides)
tstop_times = [orbital_period, orbital_period + orbital_period/100]
tstop_changes = [{}]                             # When the tstop occurs, choose how you want to alter the system
                                                 # These values represent multipliers based on original value each
                                                 # list item is a dictionary

save_data = False                                # Would you like to create a hdf5 file with the results?
folder_name = "solved_odes"                      # Save folder name

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                                                   SOLVE THE SYSTEM                                                   #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################
# System values
_, equ_radius = get_BE_equilibrium_radius(mass_r)
if equ_radius == -1:
    raise SystemExit("No proper Bonnor-Ebert equilibrium radius was found")
# Place the initial conditions and system configuration into their respective data objects
initial_conditions = InitialConditions(
    ode_init_con=[init_a1 * equ_radius, init_a1_v * equ_radius,
                  init_a2 * equ_radius, init_a2_v * equ_radius,
                  init_a3 * equ_radius, init_a3_v * equ_radius,
                  init_θ, init_θ_v,
                  init_ϕ, init_ϕ_v],
    ρ_real_over_ρ_tides=ρ_real_over_ρ_tides,
    ρ_pressure_over_ρ_tides=ρ_pressure_over_ρ_tides,
    ρ_real_over_ρ_pressure=ρ_real_over_ρ_pressure,
    mass_r=mass_r,
    equ_radius=equ_radius,
    after_tstop_params=tstop_changes
)
solver_config = ConfigParams(
    start=start_time,
    stop=stop_time,
    num_time=time_steps,
    max_steps=max_steps,
    relative_tolerance=relative_tolerance,
    absolute_tolerance=absolute_tolerance,
    enable_taylor_jump=enable_taylor_jump,
    taylor_jumps_num=taylor_jumps_num,
    taylor_jump_threshold=taylor_jump_threshold,
    enable_tstop=enable_tstop,
    tstop_times=tstop_times
)
# Compute the solution
sol_start_time = current_time()
soln, time, flag, internal_data = solution(initial_conditions=initial_conditions, solver_config=solver_config,
                                           save_data=save_data, folder_name=folder_name)
print("Time taken to compute calculation: " + str(current_time() - sol_start_time) + " seconds")

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                                                    SIMPLE PLOT                                                       #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################
print("ratios were")
print("gravity to tides: ", ρ_real_over_ρ_tides)
print("pressure to tides: ", ρ_pressure_over_ρ_tides)
print("density to pressure ratio: ", ρ_real_over_ρ_pressure)

plt.subplot(3, 2, 1)
plt.plot(time, soln[:, ODEIndex.x], label="a1", color='black')
plt.plot(time, soln[:, ODEIndex.y], label="a2")
plt.plot(time, soln[:, ODEIndex.z], label="a3", color='red')
plt.axvline(x=2 * pi * 1 * 1 * sqrt(ρ_pressure_over_ρ_tides), color='grey', alpha=0.4, ls="--", label="Orbital time")
for i in np.arange(2, 100, 1):
    if 2 * pi * i * sqrt(ρ_pressure_over_ρ_tides) < time[-1]:
        plt.axvline(x=2 * pi * i * 1 * sqrt(ρ_pressure_over_ρ_tides), color='grey', alpha=0.4, ls="--")
if enable_tstop:
    plt.axvspan(tstop_times[0], tstop_times[1], color='grey', alpha=0.4)
plt.legend(loc='upper left')
plt.xlabel("Time (T)")
plt.ylabel("Length (L)")

plt.subplot(3, 2, 2)
plt.plot(time, soln[:, ODEIndex.xdot], label="d/dt a1", color='black')
plt.plot(time, soln[:, ODEIndex.ydot], label="d/dt a2")
plt.plot(time, soln[:, ODEIndex.zdot], label="d/dt a3", color='red')
if enable_tstop:
    plt.axvspan(tstop_times[0], tstop_times[1], color='grey', alpha=0.4)
plt.legend(loc='best')
plt.xlabel("Time (T)")
plt.ylabel("Velocity (cs)")

plt.subplot(3, 2, 3)
plt.plot(time,
         (1/(soln[:, ODEIndex.x] * soln[:, ODEIndex.y] * soln[:, ODEIndex.z]) *
          (soln[0, ODEIndex.x] * soln[0, ODEIndex.y] * soln[0, ODEIndex.z])),
         label="density")
if enable_tstop:
    plt.axvspan(tstop_times[0], tstop_times[1], color='grey', alpha=0.4)
plt.legend(loc='best')
plt.xlabel("Time (T)")
plt.ylabel("Density (scaled to initial density)")

plt.subplot(3, 2, 4)
from math import degrees
plt.plot(time, [degrees(i) for i in soln[:, ODEIndex.θ]], label="θ")
plt.plot(time, [degrees(i-j) for (i, j) in zip(soln[:, ODEIndex.ϕ], soln[:, ODEIndex.θ])], label="ϕ")
plt.plot(time, [degrees(i) for i in soln[:, ODEIndex.ϕ]], label="ϕ + θ code variable")
plt.axvline(x=2 * pi * 1 * 1 * sqrt(ρ_pressure_over_ρ_tides), color='grey', alpha=0.4, ls="--", label="Orbital time")
for i in np.arange(2, 100, 1):
    if 2 * pi * i * sqrt(ρ_pressure_over_ρ_tides) < time[-1]:
        plt.axvline(x=2 * pi* i * 1 * sqrt(ρ_pressure_over_ρ_tides), color='grey', alpha=0.4, ls="--")
if enable_tstop:
    plt.axvspan(tstop_times[0], tstop_times[1], color='grey', alpha=0.4)
plt.legend(loc='best')
plt.xlabel("Time (T)")
plt.ylabel("Theta (degrees)")

plt.subplot(3, 2, 5)
plt.plot(time, soln[:, ODEIndex.θdot], label="θdot")
plt.plot(time, (soln[:, ODEIndex.ϕdot] - soln[:, ODEIndex.θdot]), label="ϕdot")
if enable_tstop:
    plt.axvspan(tstop_times[0], tstop_times[1], color='grey', alpha=0.4)
plt.legend(loc='best')
plt.xlabel("Time (T)")
plt.ylabel("d/dt theta (1/T)")

plt.subplot(3, 2, 6)
init_den = 3 * mass_r/(4 * pi * soln[0, ODEIndex.x] * soln[0, ODEIndex.y] * soln[0, ODEIndex.z])
den = 3 * mass_r/(4 * pi * soln[:, ODEIndex.x] * soln[:, ODEIndex.y] * soln[:, ODEIndex.z])
plt.axhline(y=4/ρ_real_over_ρ_pressure, linestyle="--", label="4 P/cs2 (1/init den)", color='black')
plt.axhline(y=1/ρ_real_over_ρ_pressure, linestyle="--", label="P/cs2 (1/init den)", color='blue')
plt.plot(time, den/init_den, label="Real density", color="red")
plt.axhspan(0, 4 * 1/ρ_real_over_ρ_tides* 1/(9 * 0.090068) , color='grey', alpha=0.4, label="Roche Unstable")
if enable_tstop:
    plt.axvspan(tstop_times[0], tstop_times[1], color='black', alpha=0.8)
plt.ylim(0)
plt.legend(loc='best')
plt.xlabel("Time (T)")
plt.ylabel("Rho (init Rho)")
plt.show()

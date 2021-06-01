"""
Settings used to generate figure 2.3.
"""
import matplotlib.pyplot as plt
from math import sqrt, pi
from time import time as current_time

from src.data_formats import InitialConditions, ConfigParams
from src.utils import get_BE_equilibrium_radius, get_BE_mass_0to5sqrt5o16, get_BE_mass_1to4_ratio
from src.solve.solution import solution

from src.utils import ODEIndex

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                                                SET SYSTEM PARAMETERS                                                 #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################
# System params
# Set system values
ratio = 1                                    # Specify a ratio for the Bonnor-Ebert mass. e.g. 10% = 0.1
method = "manual"                            # Specify method for initial conditions.
# Calculate required values
ρ_real_over_ρ_pressure = get_BE_mass_1to4_ratio(ratio)  # Should be 3.999999999998998

# ## ### #### ##### ###### ####### manual ####### ###### ##### #### ### ## #
# Set system values, Note set manually init_ai
ρ_real_over_ρ_tides = 1000000000000.0
# Calculate required values
ρ_pressure_over_ρ_tides = ρ_real_over_ρ_tides/ρ_real_over_ρ_pressure  # Should be 250000000000.06262
mass_r = get_BE_mass_0to5sqrt5o16(ρ_real_over_ρ_pressure)  # Should be 0.6987712429686843
_, equ_radius = get_BE_equilibrium_radius(mass_r)  # Should be 0.559016991193415
if equ_radius == -1:
    raise SystemExit("No Bonnor-Ebert equilibrium radius was found")

# Equation initial conditions
init_a1 = 1.005                                 # Initial a1 axis length of ellipsoid, a1 is usually scaled to 1.005
init_a2 = 0.995                                 # Initial a2 axis length, a2 is usually scaled to 0.995
init_a3 = 1.0                                   # Initial a3 axis length, a3 is usually scaled to 1.000
init_θ = 0                                      # Initial θ angle in radians
init_ϕ = 0                                      # Initial ϕ angle in radians
init_a1_v = 0                                   # Initial velocity of a1 axis, scaled by equ. radius
init_a2_v = 0                                   # Initial velocity of a2 axis, scaled by equ. radius
init_a3_v = 0                                   # Initial velocity of a3 axis, scaled by equ. radius
init_θ_v = 0                                    # Initial velocity of the θ rotation angle
init_ϕ_v = 0                                    # Initial velocity of the ϕ rotation angle

start_time = 0                                  # Time to start the simulation
stop_time = 3                                   # Time to stop simulation in seconds - will be scaled by time unit
time_steps = 10000                        # Number of time steps to solve during the simulation
max_steps = 10000000                              # Max number of iterations the solver may take to solve a timestep
relative_tolerance = 1e-12                      # Relative tolerance of the solver
absolute_tolerance = 1e-12                      # Absolute tolerance of the solver
enable_taylor_jump = False                      # Choice to jump axis length changes with a taylor series approximation
taylor_jumps_num = 1                            # how many taylor jumps the system is allowed.
taylor_jump_threshold = 0.005                   # Distance between a1 and a2 axis at which to employ taylor series
enable_tstop = False                            # Choice to stop the simulation at certain times
orbital_period = 2 * pi * sqrt(ρ_pressure_over_ρ_tides)
tstop_times = [1.01, 1.]
tstop_changes = [{}]                            # When the tstop occurs, choose how you want to alter the system
                                                # These values represent multipliers based on original value each
                                                # list item is a dictionary

save_data = False                                # Would you like to create a hdf5 file with the results?
folder_name = "solutions"                        # Save folder name

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                                                   SOLVE THE SYSTEM                                                   #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################
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

print("Initial axis lengths", soln[0, ODEIndex.x], soln[0, ODEIndex.y], soln[0, ODEIndex.z])
print("initial conditions", initial_conditions, solver_config)

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                                                    SIMPLE PLOT                                                       #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################
print("Density ratios were:")
print("gravity to tides: ", ρ_real_over_ρ_tides)
print("pressure to tides: ", ρ_pressure_over_ρ_tides)
print("density to pressure ratio: ", ρ_real_over_ρ_pressure)
plt.subplot(2, 2, 1)
plt.plot(time, soln[:, ODEIndex.x], label="a1", color='red')
plt.plot(time, soln[:, ODEIndex.y], label="a2", color='green')
plt.xlim(0, 3)
plt.legend(loc='upper right')
plt.xlabel("Time (T)")
plt.ylabel("Length (L)")


plt.subplot(2, 2, 2)
from math import degrees
plt.plot(time, [degrees(i) for i in soln[:, ODEIndex.θ]], label="θ", color='red')
plt.plot(time, [degrees(i-j) for (i, j) in zip(soln[:, ODEIndex.ϕ], soln[:, ODEIndex.θ])], label="ϕ", color='green')
plt.plot(time, [degrees(i) for i in soln[:, ODEIndex.ϕ]], label="ϕ + θ code variable", color='blue')
plt.legend(loc='upper left')
plt.xlabel("Time (T)")
plt.ylabel("Angle (deg)")
plt.axhline(y=90, ls="--", color='grey', alpha=0.4, lw=2)
plt.axhline(y=180, ls="--", color='grey', alpha=0.4, lw=2)
plt.axhline(y=270, ls="--", color='grey', alpha=0.4, lw=2)
plt.axhline(y=-90, ls="--", color='grey', alpha=0.4)
plt.axhline(y=-180, ls="--", color='grey', alpha=0.4)
plt.axhline(y=-270, ls="--", color='grey', alpha=0.4)
plt.xlim(0, 3)
plt.ylim(-280, 280)


plt.subplot(2, 2, 3)
plt.plot(time, soln[:, ODEIndex.xdot], label="d/dt a1", color='red')
plt.plot(time, soln[:, ODEIndex.ydot], label="d/dt a2", color='green')
plt.legend(loc='upper right')
plt.xlabel("Time (T)")
plt.ylabel("Velocity (cs)")
plt.xlim(0, 3)

plt.subplot(2, 2, 4)
plt.plot(time, soln[:, ODEIndex.x], label="a1", color='red')
plt.plot(time, soln[:, ODEIndex.y], label="a2", color='green')
plt.legend(loc='upper right')
plt.xlim(0.502, 0.512)
plt.ylim(0.5590, 0.5591)
plt.xlabel("Time (T)")
plt.ylabel("Length (L)")

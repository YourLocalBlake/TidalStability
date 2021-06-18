# dirty test to call the main function, will fail unless sundials and odes is
# installed and working correctly.
from math import sqrt, pi
from tidal_stability.utils import (
    get_BE_equilibrium_radius,
    get_BE_mass_0to5sqrt5o16,
    get_BE_mass_1to4_ratio,
)
from tidal_stability.data_formats import InitialConditions, ConfigParams
from tidal_stability.solve.solution import solution

ratio = 0.9
ρ_real_over_ρ_tides = 25

ρ_real_over_ρ_pressure = get_BE_mass_1to4_ratio(ratio)
ρ_pressure_over_ρ_tides = ρ_real_over_ρ_tides / ρ_real_over_ρ_pressure
mass_r = get_BE_mass_0to5sqrt5o16(ρ_real_over_ρ_pressure)

init_a1 = 1.005
init_a2 = 0.995
init_a3 = 1.000
init_θ = 0
init_ϕ = 0
init_a1_v = 0
init_a2_v = 0
init_a3_v = 0
init_θ_v = 0
init_ϕ_v = 0

start_time = 0
stop_time = 0.1
time_steps = 10e3
max_steps = 10000
relative_tolerance = 1e-7
absolute_tolerance = 1e-7
enable_taylor_jump = False
taylor_jumps_num = 1
taylor_jump_threshold = 0.005
enable_tstop = False
orbital_period = 2 * pi * sqrt(ρ_pressure_over_ρ_tides)
tstop_times = [orbital_period, orbital_period + orbital_period / 100]
tstop_changes = [{}]

save_data = False
folder_name = "solved_odes"
_, equ_radius = get_BE_equilibrium_radius(mass_r)
if equ_radius == -1:
    raise SystemExit("No proper Bonnor-Ebert equilibrium radius was found")
initial_conditions = InitialConditions(
    ode_init_con=[
        init_a1 * equ_radius,
        init_a1_v * equ_radius,
        init_a2 * equ_radius,
        init_a2_v * equ_radius,
        init_a3 * equ_radius,
        init_a3_v * equ_radius,
        init_θ,
        init_θ_v,
        init_ϕ,
        init_ϕ_v,
    ],
    ρ_real_over_ρ_tides=ρ_real_over_ρ_tides,
    ρ_pressure_over_ρ_tides=ρ_pressure_over_ρ_tides,
    ρ_real_over_ρ_pressure=ρ_real_over_ρ_pressure,
    mass_r=mass_r,
    equ_radius=equ_radius,
    after_tstop_params=tstop_changes,
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
    tstop_times=tstop_times,
)


class Testclass:
    def test_main(self):
        soln, time, flag, internal_data = solution(
            initial_conditions=initial_conditions,
            solver_config=solver_config,
            save_data=save_data,
            folder_name=folder_name,
        )
        if soln[0, 0]:  # First index of the x dim.
            assert 1

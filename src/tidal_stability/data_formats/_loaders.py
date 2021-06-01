"""
Load phy5 files
"""
import h5py

from tidal_stability.data_formats import Solution, InitialConditions, ConfigParams, InternalData


def solution_loader(override=False, file_name=None, folder_name="solved_odes"):
    """
    Load a hdf5 file containing a solved ODE
    Returns the 4 data objects
    """

    file = str(folder_name)+"/TidalStability_"+str(file_name)+".hdf5"

    if override:
        file = override

    with h5py.File(file, "r") as f:
        solution = Solution(
            solution=f["solution"]["solution"][:],
            times=f["solution"]["times"][:],
            flags=f["solution"]["flags"][:],
            tstop_times=f["solution"].attrs["tstop_times"],
            jump_times=f["solution"].attrs["jump_times"]
        )

        config_params = ConfigParams(
            start=f["solver_config"].attrs["start"],
            stop=f["solver_config"].attrs["stop"],
            num_time=f["solver_config"].attrs["num_time"],
            max_steps=f["solver_config"].attrs["max_steps"],
            relative_tolerance=f["solver_config"].attrs["relative_tolerance"],
            absolute_tolerance=f["solver_config"].attrs["absolute_tolerance"],
            enable_taylor_jump=f["solver_config"].attrs["taylor_jump_enabled"],
            taylor_jumps_num=f["solver_config"].attrs["taylor_jumps_num"],
            taylor_jump_threshold=f["solver_config"].attrs["taylor_jump_threshold"],
            enable_tstop=f["solver_config"].attrs["enable_tstop"],
            tstop_times=f["solver_config"].attrs["tstop_times"]
        )

        internal_data = InternalData(
            params=f["internal_data"]["params"][:],
            derivs=f["internal_data"]["derivs"][:],
            times=f["internal_data"]["times"][:],
            a1=f["internal_data"]["a1"][:],
            a1dot=f["internal_data"]["a1dot"][:],
            a2=f["internal_data"]["a2"][:],
            a2dot=f["internal_data"]["a2dot"][:],
            a3=f["internal_data"]["a3"][:],
            a3dot=f["internal_data"]["a3dot"][:],
            θ=f["internal_data"]["θ"][:],
            θdot=f["internal_data"]["θdot"][:],
            ϕ=f["internal_data"]["ϕ"][:],
            ϕdot=f["internal_data"]["ϕdot"][:],
            problems=f["internal_data"].attrs["problems"]
        )

        initial_conditions = InitialConditions(
            ode_init_con=f["initial_conditions"]["ode_initial_conditions"][:],
            ρ_real_over_ρ_tides=f["initial_conditions"].attrs["ρ_real_over_ρ_tides"],
            ρ_pressure_over_ρ_tides=f["initial_conditions"].attrs["ρ_pressure_over_ρ_tides"],
            ρ_real_over_ρ_pressure=f["initial_conditions"].attrs["ρ_real_over_ρ_pressure"],
            mass_r=f["initial_conditions"].attrs["mass_r"],
            equ_radius=f["initial_conditions"].attrs["equ_radius"],
        )

    return solution, initial_conditions, config_params, internal_data

"""
prepares the data classes to be dumped to hdf5 files
"""

import h5py
import arrow
from numpy import array


def create_results_file(solution_dump, internal_data_dump, init_con_dump, solver_config_dump, folder_name):
    """
    Save the results from a solution run in a hdf5 file
    inputs are the 4 data objects/classes
    """
    # todo: far future: implement naming convention based on system parameters not time solved at.
    if len(solution_dump.times) > 1:  # Require a solution exists.
        # with h5py.File("{}/TidalStability_".format(folder_name) + "sg_tides_{}_sg_p_{}_mr_{}".format(
        #     init_con_dump.ρ_real_over_ρ_tides, init_con_dump.ρ_real_over_ρ_pressure,
        #     init_con_dump.mass_r * 1/(5/16 * sqrt(5))) + ".hdf5", "x") as f:
        with h5py.File("{}/TidalStability_".format(folder_name) + str(arrow.now().format("YYYY-MM-DD--HH-mm")), "x") as f:
            _solution_dumper(f.create_group("solution"), solution_dump, solver_config_dump)
            _initial_conditions(f.create_group("initial_conditions"), init_con_dump, solver_config_dump)
            _config_dumper(f.create_group("solver_config"), solver_config_dump)
            _internal_data_dumper(f.create_group("internal_data"), internal_data_dump)
            f.attrs["date_solved"] = [str(arrow.now().format("YYYY-MM-DD--HH-mm"))]
            # print("Solution saved as '/{}/TidalStability_".format(folder_name)
            #       + "sg_tides_{}_sg_p_{}_mr_{}".format(init_con_dump.ρ_real_over_ρ_tides,
            #       init_con_dump.ρ_real_over_ρ_pressure, init_con_dump.mass_r * 1/(5/16 * sqrt(5))) + ".hdf5'")
            print("Solution saved as '/{}/TidalStability_".format(folder_name)
                  + str(arrow.now().format("YYYY-MM-DD--HH-mm")) + ".hdf5")
            # Doesn't return anything, just saves the file.
    else:
        print("No properly formatted solution was given. No hdf5 file created.")


def _solution_dumper(grp, solution, solver_config):
    """
    Solution to the ODE dumper
    :param: grp: hdf5 group
    :param: solution: A solution data class instance i.e. Solution class
    """
    grp.create_dataset("times", data=solution.times)
    grp.create_dataset("solution", data=solution.solution)
    grp.create_dataset("flags", data=solution.flags)
    if solver_config.enable_tstop:
        try:
            grp.attrs["tstop_times"] = [time for time in solution.tstop_times]
        except TypeError:
            grp.attrs["tstop_times"] = ["No_tstops"]
    else:
        grp.attrs["tstop_times"] = ["NOT_ENABLED"]
    if solver_config.enable_taylor_jump:
        try:
            grp.attrs["jump_times"] = [i.encode('utf8') for i in solution.jump_times]
        except TypeError:
            grp.attrs["jump_times"] = ["No_Jumps"]
    else:
        grp.attrs["jump_times"] = ["NOT_ENABLED"]
    return grp


def _initial_conditions(grp, init_con, solver_config):
    """
    Initial conditions dumper
    """

    grp.create_dataset("ode_initial_conditions", data=init_con.ode_init_con)
    grp.attrs["ρ_real_over_ρ_tides"] = init_con.ρ_real_over_ρ_tides
    grp.attrs["ρ_pressure_over_ρ_tides"] = init_con.ρ_pressure_over_ρ_tides
    grp.attrs["ρ_real_over_ρ_pressure"] = init_con.ρ_real_over_ρ_pressure
    grp.attrs["mass_r"] = init_con.mass_r
    grp.attrs["equ_radius"] = init_con.equ_radius
    if solver_config.enable_tstop:
        grp.attrs["after_tstop_params"] = str(init_con.after_tstop_params)
    else:
        grp.attrs["after_tstop_params"] = str([])

    return grp


def _config_dumper(grp, config):
    """
    Configuration dumper
    """
    grp.attrs["start"] = config.start
    grp.attrs["stop"] = config.stop
    grp.attrs["num_time"] = config.num_time
    grp.attrs["max_steps"] = config.max_steps
    grp.attrs["relative_tolerance"] = config.relative_tolerance
    grp.attrs["absolute_tolerance"] = config.absolute_tolerance
    grp.attrs["taylor_jump_enabled"] = config.enable_taylor_jump
    grp.attrs["taylor_jumps_num"] = config.taylor_jumps_num
    grp.attrs["taylor_jump_threshold"] = config.taylor_jump_threshold
    grp.attrs["enable_tstop"] = config.enable_tstop
    grp.attrs["tstop_times"] = config.tstop_times
    return grp


def _internal_data_dumper(grp, internal_data):
    """
    Internal data dumper
    """
    grp.create_dataset("params", data=internal_data.params)
    grp.create_dataset("derivs", data=internal_data.derivs)
    grp.create_dataset("times", data=internal_data.times)
    grp.create_dataset("a1", data=internal_data.a1)
    grp.create_dataset("a1dot", data=internal_data.a1dot)
    grp.create_dataset("a2", data=internal_data.a2)
    grp.create_dataset("a2dot", data=internal_data.a2dot)
    grp.create_dataset("a3", data=internal_data.a3)
    grp.create_dataset("a3dot", data=internal_data.a3dot)
    grp.create_dataset("θ", data=internal_data.θ)
    grp.create_dataset("θdot", data=internal_data.θdot)
    grp.create_dataset("ϕ", data=internal_data.ϕ)
    grp.create_dataset("ϕdot", data=array(internal_data.ϕdot))
    grp.attrs["problems"] = [i.encode('utf8') for i in internal_data.problems]
    return grp

"""
File which defines the ODE system and creates a callable solver.
All work is done in dimensionless units unless stated
"""
from os import mkdir
from os.path import isdir
from copy import deepcopy
from numpy import copy, linspace, concatenate, asarray, append as npappend
from mpmath import mp

from scikits.odes import ode
from scikits.odes.sundials import CVODESolveFailed, CVODESolveFoundRoot, CVODESolveReachedTSTOP
from scikits.odes.sundials.cvode import StatusEnum

from src.utils import ODEIndex, EllipIndex
from src.data_formats import InternalData, Solution, create_results_file
from src.solve.utils import xy_close_stop_generator, get_jump_amount, ontstop_cont, ontstop_stop
from src.solve.geometry import get_Ax, get_Ay, get_Az
from src.solve.deriv_funcs import (deriv_x_func, deriv_xdot_func,
                                   deriv_y_func, deriv_ydot_func,
                                   deriv_z_func, deriv_zdot_func,
                                   deriv_θ_func, deriv_θdot_func,
                                   deriv_ϕ_func, deriv_ϕθdot_func,
                                   deriv_ϕdot_func)


def ode_system(*, a1, a2, a3, mass_r, ρ_pressure_over_ρ_tides, ρ_real_over_ρ_pressure, store_internal=True):
    """
    Create the mathematical ODE model for the tri-axial ellipsoid.
    Returns the sundials ode model which will be solved.
    """
    # Constants of equations

    ρ_real_over_ρ_pressure = ρ_real_over_ρ_pressure * (a1 * a2 * a3)

    if store_internal:
        internal_data = InternalData()
        times_list = internal_data.times
        params_list = internal_data.params
        derivs_list = internal_data.derivs
        a1_list = internal_data.a1
        a2_list = internal_data.a2
        a3_list = internal_data.a3
        θ_list = internal_data.θ
        ϕ_list = internal_data.ϕ
        a1dot_list = internal_data.a1dot
        a2dot_list = internal_data.a2dot
        a3dot_list = internal_data.a3dot
        θdot_list = internal_data.θdot
        ϕdot_list = internal_data.ϕdot
        problems = internal_data.problems
    else:
        internal_data = None

    def rhs_equation(time, params, derivs):
        """
        RHS of the ODE equations
        Inputs are passed by Sundials.
        """
        # Specify the variables
        x = params[ODEIndex.x]
        y = params[ODEIndex.y]
        z = params[ODEIndex.z]
        θ = params[ODEIndex.θ]
        ϕ = params[ODEIndex.ϕ]
        xdot = params[ODEIndex.xdot]
        ydot = params[ODEIndex.ydot]
        zdot = params[ODEIndex.zdot]
        θdot = params[ODEIndex.θdot]
        ϕdot = params[ODEIndex.ϕdot]

        # Symbols and time-dependent-constants for equations
        Ai = [get_Ax(x=x, y=y, z=z), get_Ay(x=x, y=y, z=z), get_Az(x=x, y=y, z=z)]
        ρ_real_over_ρ_pressure_updating = ρ_real_over_ρ_pressure/(x * y * z)
        # Physical checks:
        if x <= 0.01 or y <= 0.01 or z <= 0.01:
            if store_internal:
                problems.append("-1. Lengths about to negative, lengths were {}, {}, {}".format(x, y, z))
                print("Lengths about to go negative, they are {}. Stopping.".format(x, y, z))
            return -1

        # First order derivative functions
        deriv_x, x_i = deriv_x_func(xdot=xdot)
        deriv_y, y_i = deriv_y_func(ydot=ydot)
        deriv_z, z_i = deriv_z_func(zdot=zdot)
        deriv_θ, θ_i = deriv_θ_func(θdot=θdot)
        deriv_ϕ, ϕ_i = deriv_ϕ_func(ϕdot=ϕdot)

        # Second order derivative functions
        deriv_xdot, xdot_i = deriv_xdot_func(x=x, y=y, z=z, θ=θ, θdot=θdot, ϕdot=ϕdot-θdot, mass_r=mass_r,
                                             A1=Ai[EllipIndex.x],
                                             ρ_real_over_ρ_pressure=ρ_real_over_ρ_pressure_updating,
                                             ρ_pressure_over_ρ_tides=ρ_pressure_over_ρ_tides)
        deriv_ydot, ydot_i = deriv_ydot_func(x=x, y=y, z=z, θ=θ, θdot=θdot, ϕdot=ϕdot-θdot,  mass_r=mass_r,
                                             A2=Ai[EllipIndex.y],
                                             ρ_real_over_ρ_pressure=ρ_real_over_ρ_pressure_updating,
                                             ρ_pressure_over_ρ_tides=ρ_pressure_over_ρ_tides)
        deriv_zdot, zdot_i = deriv_zdot_func(x=x, y=y, z=z,  mass_r=mass_r, A3=Ai[EllipIndex.z],
                                             ρ_real_over_ρ_pressure=ρ_real_over_ρ_pressure_updating,
                                             ρ_pressure_over_ρ_tides=ρ_pressure_over_ρ_tides)
        deriv_θdot, θdot_i = deriv_θdot_func(x=x, xdot=xdot, y=y, ydot=ydot, θ=θ, θdot=θdot, ϕdot=ϕdot-θdot,
                                             ρ_pressure_over_ρ_tides=ρ_pressure_over_ρ_tides)
        deriv_ϕdot, ϕdot_i = deriv_ϕθdot_func(x=x, xdot=xdot, y=y, ydot=ydot, θ=θ, θdot=θdot, ϕdot=ϕdot-θdot,
                                              ρ_pressure_over_ρ_tides=ρ_pressure_over_ρ_tides)

        # Solve
        derivs[ODEIndex.x] = deriv_x
        derivs[ODEIndex.y] = deriv_y
        derivs[ODEIndex.z] = deriv_z
        derivs[ODEIndex.θ] = deriv_θ
        derivs[ODEIndex.ϕ] = deriv_ϕ
        derivs[ODEIndex.xdot] = deriv_xdot
        derivs[ODEIndex.ydot] = deriv_ydot
        derivs[ODEIndex.zdot] = deriv_zdot
        derivs[ODEIndex.θdot] = deriv_θdot
        derivs[ODEIndex.ϕdot] = deriv_ϕdot

        if store_internal:
            params_list.append(copy(params))
            derivs_list.append(copy(derivs))
            times_list.append(copy(time))
            a1_list.append(copy(x_i))
            a2_list.append(copy(y_i))
            a3_list.append(copy(z_i))
            θ_list.append(copy(θ_i))
            ϕ_list.append(copy(ϕ_i))
            a1dot_list.append(copy(xdot_i))
            a2dot_list.append(copy(ydot_i))
            a3dot_list.append(copy(zdot_i))
            θdot_list.append(copy(deriv_θdot))
            ϕdot_list.append(copy(deriv_ϕdot))

    return rhs_equation, internal_data


def calc_numerical_solution(*, initial_conditions, solver_config, time, tstop):
    """
    Run the sundials ODE solver on the set of differential equations
    """
    # initialise the differential equation model
    system, internal_data = ode_system(
       a1=initial_conditions.ode_init_con[0],
       a2=initial_conditions.ode_init_con[2],
       a3=initial_conditions.ode_init_con[4],
       mass_r=initial_conditions.mass_r,
       ρ_pressure_over_ρ_tides=initial_conditions.ρ_pressure_over_ρ_tides,
       ρ_real_over_ρ_pressure=initial_conditions.ρ_real_over_ρ_pressure
       )

    # If taylor jump is selected, then add the additional params to the ode solver to perform the action
    tay_args = {}
    # If solver_config.enable_taylor_jump:
    if solver_config.enable_taylor_jump:
        tay_args["rootfn"] = xy_close_stop_generator(solver_config.taylor_jump_threshold)
        tay_args["nr_rootfns"] = 1

    # If time stop is selected, then set the solver to stop at this timepoint
    ontstop_func = ontstop_stop() if solver_config.enable_tstop else ontstop_cont()

    # Create the sundials system
    solver = ode(
        'cvode', system,
        old_api=False,
        validate_flags=True,
        rtol=solver_config.relative_tolerance,
        atol=solver_config.absolute_tolerance,
        max_steps=solver_config.max_steps,
        tstop=tstop,
        ontstop=ontstop_func,
        **tay_args
    )

    # and solve it.
    try:
        soln = solver.solve(time, initial_conditions.ode_init_con)
    except CVODESolveFailed as e:
        soln = e.soln
        print("Solver: FAILED at time {} with conditions {}".format(soln.values.t[-1], soln.values.y[-1, :]))
    except CVODESolveFoundRoot as e:
        soln = e.soln
        print("Solver: FOUND ROOT at time {} with conditions {}".format(soln.values.t[-1], soln.values.y[-1, :]))
    except CVODESolveReachedTSTOP as e:
        soln = e.soln
        print("Solver: TIME STOP at time {}".format(soln.values.t[-1]))
    # except:  # Bare except for problems with division by zero
    #     raise SystemExit("Something went horribly wrong when trying to solve the ODE. Exiting")

    return soln, internal_data


def calc_xy_singularity(*, current_sol, jump_amount, system_config):
    """
    Preform a taylor series expansion of the position and velocity at the input time.
    """
    # Get the most recent time and list of values.
    continue_time = current_sol.values.t[-1]
    continue_soln = current_sol.values.y[-1]
    # time period to jump over
    dt = jump_amount

    # Split equations into components
    x = continue_soln[ODEIndex.x]
    y = continue_soln[ODEIndex.y]
    z = continue_soln[ODEIndex.z]
    θ = continue_soln[ODEIndex.θ]
    ϕ = continue_soln[ODEIndex.ϕ]
    xdot = continue_soln[ODEIndex.xdot]
    ydot = continue_soln[ODEIndex.ydot]
    zdot = continue_soln[ODEIndex.zdot]
    θdot = continue_soln[ODEIndex.θdot]
    ϕdot = continue_soln[ODEIndex.ϕdot]

    # Obtain other components needed to calculate the derivatives
    Ai = [get_Ax(x=x, y=y, z=z), get_Ay(x=x, y=y, z=z), get_Az(x=x, y=y, z=z)]
    ρ_real_over_ρ_pressure_updating = current_sol.ρ_real_over_ρ_pressure * (x * y * z)

    # Numerically calculate the first and second order derivatives at the time point using the previous timestep values
    deriv_xdot, _ = deriv_xdot_func(x=x, y=y, z=z, θ=θ, θdot=θdot, ϕdot=ϕdot-θdot, A1=Ai[EllipIndex.x],
                                    ρ_real_over_ρ_pressure=ρ_real_over_ρ_pressure_updating,
                                    ρ_pressure_over_ρ_tides=current_sol.ρ_pressure_over_ρ_tides)
    deriv_ydot, _ = deriv_ydot_func(x=x, y=y, z=z, θ=θ, θdot=θdot, ϕdot=ϕdot-θdot, A2=Ai[EllipIndex.y],
                                    ρ_real_over_ρ_pressure=ρ_real_over_ρ_pressure_updating,
                                    ρ_pressure_over_ρ_tides=current_sol.ρ_pressure_over_ρ_tides)
    deriv_zdot, _ = deriv_zdot_func(x=x, y=y, z=z, A3=Ai[EllipIndex.z],
                                    ρ_real_over_ρ_pressure=ρ_real_over_ρ_pressure_updating,
                                    ρ_pressure_over_ρ_tides=current_sol.ρ_pressure_over_ρ_tides)
    deriv_θdot, _ = deriv_θdot_func(x=x, xdot=xdot, y=y, ydot=ydot, θ=θ, θdot=θdot, ϕdot=ϕdot-θdot,
                                    ρ_pressure_over_ρ_tides=current_sol.ρ_pressure_over_ρ_tides)
    deriv_ϕdot, _ = deriv_ϕθdot_func(x=x, xdot=xdot, y=y, ydot=ydot, θ=θ, θdot=θdot, ϕdot=ϕdot-θdot,
                                     ρ_pressure_over_ρ_tides=current_sol.ρ_pressure_over_ρ_tides)

    # Calculate the values of the jump
    x = x + xdot * dt + 1/2 * deriv_xdot * dt**2
    y = y + ydot * dt + 1/2 * deriv_ydot * dt**2
    z = z + zdot * dt + 1/2 * deriv_zdot * dt**2
    θ = θ + θdot * dt + 1/2 * deriv_θdot * dt**2
    ϕ = ϕ + ϕdot * dt + 1/2 * deriv_ϕdot * dt**2
    xdot = xdot + deriv_xdot * dt
    ydot = ydot + deriv_ydot * dt
    zdot = zdot + deriv_zdot * dt
    θdot = θdot + deriv_θdot * dt
    ϕdot = ϕdot + deriv_ϕdot * dt

    return continue_time + dt, [x, xdot, y, ydot, z, zdot, θ, θdot, ϕ, ϕdot]


def solution(*, initial_conditions, solver_config, save_data, folder_name):
    """
    Compute the solution to the model given the input initial conditions and solver configuration
    Saves the solution file.
    :return: List of lists: solution, times, internal_data object
    """
    mp.dps = 50

    # Initialise the time over which to solve the system
    time = linspace(solver_config.start, solver_config.stop, solver_config.num_time)
    time_length_passed = solver_config.num_time

    # Compute the solution
    if not solver_config.enable_taylor_jump and not solver_config.enable_tstop:  # No Taylor jump or tstop
        print("Solving the ODE system!")
        soln, internal_data = calc_numerical_solution(
            initial_conditions=initial_conditions,
            solver_config=solver_config,
            time=time,
            tstop=0)
        soln_y = soln.values.y
        soln_t = soln.values.t
        soln_flag = [soln.flag]
        jump_times = []

    elif solver_config.enable_taylor_jump or solver_config.enable_tstop:  # Either Taylor jumps or tstops

        # Checks to make sure everything required for either tay jumps or tstops is present.
        if solver_config.enable_tstop:
            if len(initial_conditions.after_tstop_params) != len(solver_config.tstop_times):
                raise RuntimeError("Input Error: Amount of tstops ({}) != amount of new params ({})".format(
                    initial_conditions.after_tstop_params, solver_config.tstop_times))
            if solver_config.tstop_times[-1] > solver_config.stop:
                raise RuntimeError("Input Error: Selected to use tstops after simulation will end")
        if solver_config.enable_taylor_jump:
            if solver_config.taylor_jump_threshold <= 0:
                raise RuntimeError("Input Error: Selected non-positive ({}) taylor jump threshold".format(
                    solver_config.taylor_jump_threshold))

        # Initialise the data objects to solve the solution where we can jump several times
        soln_y = None
        soln_t = None
        soln_flag = None
        internal_data = None
        jump_times = []

        # Set the first round of initial conditions for the ode
        current_soln_initial_conditions = deepcopy(initial_conditions)

        # Create the tstops and taylor jumps
        tstop_times_remaining = solver_config.tstop_times.copy()
        tstop_times_remaining.append(0)  # Add a zero so that the final run will not stop until specified end time
        tay_jumps_remaining = solver_config.taylor_jumps_num

        print("Solving the ODE system!")
        while True:

            # Compute solution
            current_soln, current_internal_data = calc_numerical_solution(
                initial_conditions=current_soln_initial_conditions,
                solver_config=solver_config,
                time=time,
                tstop=tstop_times_remaining[0])

            # save solution
            if soln_y is None:
                soln_y = current_soln.values.y
                soln_t = current_soln.values.t
                soln_flag = [current_soln.flag]
                internal_data = current_internal_data

            else:
                soln_y = concatenate((soln_y, current_soln.values.y))
                soln_t = concatenate((soln_t, current_soln.values.t))
                soln_flag = soln_flag + [current_soln.flag]
                internal_data = internal_data + current_internal_data

            # Check why the solver stopped.
            if current_soln.flag == StatusEnum.SUCCESS:  # Solver Completed
                print("Solver succeeded.")
                break

            elif current_soln.flag == StatusEnum.ROOT_RETURN:  # Solver found root, i.e. theta double dot singularity
                print("Solver: attempting to preform Taylor jump procedure.")
                if not solver_config.enable_taylor_jump:
                    print("Solver stopped because it found the root but Taylor jump is disabled")
                    break
                else:
                    if tay_jumps_remaining != 0:
                        print("Performing taylor jump at {}".format(current_soln.values.t[-1]))
                        jump_amount = get_jump_amount(
                            x=current_soln.values.y[-1, ODEIndex.x],
                            xdot=current_soln.values.y[-1, ODEIndex.xdot],
                            y=current_soln.values.y[-1, ODEIndex.y],
                            ydot=current_soln.values.y[-1, ODEIndex.ydot],
                            override=False  # can override with manual number. e.g. 3e-3
                        )
                        post_jump_time, post_jump_sol = calc_xy_singularity(
                            current_sol=current_soln,
                            jump_amount=jump_amount,
                            system_config=initial_conditions
                        )
                        # Save the jump data
                        soln_y = concatenate((soln_y, asarray([post_jump_sol])), axis=0)
                        soln_t = npappend(soln_t, post_jump_time)
                        soln_flag = soln_flag + [current_soln.flag]
                        jump_times.append("Solver: Successfully performed Taylor jump at time {}".format(
                            current_soln.values.t[-1]))
                        tay_jumps_remaining -= 1

                        # Check if we taylor jumped over the specified stopping time
                        if post_jump_time > solver_config.stop:
                            print("Solver: Jumped over the wanted stop time")
                            break
                        # Check if Taylor series jumped over a tstop time.
                        if solver_config.enable_tstop:
                            if tstop_times_remaining != [0]:
                                if post_jump_sol > tstop_times_remaining[0]:
                                    print("Solver: Taylor series jumped over a tstop time. I will continue anyway.")
                        # Reset the initial conditions of the solver to the post-jump conditions
                        current_soln_initial_conditions.ode_init_con = post_jump_sol
                        time_length_passed = time_length_passed - len(current_soln.values.t)
                        time = linspace(post_jump_time, solver_config.stop, time_length_passed)

                    else:
                        print("The solver wanted to preform a Taylor Jump but you run out of allowed Jumps. "
                              "Try increasing the amount of allowed jumps")
                        break

            elif current_soln.flag == StatusEnum.TSTOP_RETURN:  # Solver got to the required stop time.
                print("Solver: attempting to preform tstop procedure.")
                if not solver_config.enable_tstop:
                    print("Solver wanted to stop because it reached tstop but tstop is disabled.")
                # Change the initial conditions for the new run
                current_soln_initial_conditions.ode_init_con = current_soln.values.y[-1, :]
                time_length_passed = time_length_passed - len(current_soln.values.t)
                time = linspace(current_soln.values.t[-1], solver_config.stop, time_length_passed)
                # Update the initial conditions that the solver will call after tstop.
                print("vals going in are", current_soln_initial_conditions)
                for key, val in current_soln_initial_conditions.after_tstop_params[0].items():
                    # Set the new initial conditions to be the updated initial conditions
                    print("key, val", key, val)
                    print("inital ones ", getattr(initial_conditions, key))
                    print("thing before", getattr(current_soln_initial_conditions, key))
                    if key == "ode_init_con":
                        setattr(current_soln_initial_conditions, key, val * getattr(current_soln_initial_conditions, key))
                    else:
                        setattr(current_soln_initial_conditions, key, val * getattr(initial_conditions, key))
                    print("thing after", getattr(current_soln_initial_conditions, key))
                current_soln_initial_conditions.after_tstop_params.pop(0)
                # Change the tstop for the following run.
                if tstop_times_remaining == [0]:
                    print("Solver has completed all the required tstops. Will now compute the solution to end of time")
                else:
                    tstop_times_remaining.pop(0)
                    print("Solver: preformed the tstop procedure")

            elif current_soln.flag == StatusEnum.WARNING:
                print("The solver completed solving the solution but something unusable happened")
                print("The message was {}".format(current_soln.message))

            elif current_soln.flag < 0:
                print("The solver failed.")
                print("Flag = {}".format(current_soln.flag))
                print("Message = {}".format(current_soln.message))
                break

            else:
                raise SystemExit("The solver returned a unknown flag. Exiting")

    else:
        print("Could not figure out how you wanted to solve the system.")
        raise SystemExit

    # Create and save the results file
    if save_data:
        if not isdir(folder_name):
            print("No directory '{}' existed. Creating directory.".format(folder_name))
            try:
                mkdir(folder_name)
            except OSError:
                print("Could not create the directory {}.".format(folder_name))
                print("Fix directory or set save data to False")
                raise SystemExit
            else:
                print("Directory created")

        create_results_file(
            Solution(solution=soln_y, times=soln_t, flags=soln_flag, tstop_times=solver_config.tstop_times,
                     jump_times=jump_times),
            internal_data,
            initial_conditions,
            solver_config,
            folder_name,
        )

    return soln_y, soln_t, soln_flag, internal_data

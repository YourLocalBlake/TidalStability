"""
Define the data structures used.
"""
import attr
from numpy import asarray, concatenate

from src.utils import ODEIndex


@attr.s
class InitialConditions:
    """
    Datastructure for the initial conditions of the system and ODE
    """
    ρ_real_over_ρ_tides = attr.ib()
    ρ_pressure_over_ρ_tides = attr.ib()
    ρ_real_over_ρ_pressure = attr.ib()
    mass_r = attr.ib()
    equ_radius = attr.ib()
    ode_init_con = attr.ib(default=attr.Factory(list))
    after_tstop_params = attr.ib(default=attr.Factory(list))
    @ode_init_con.validator
    def check(self, attribute, values):
        if len(values) != 10:
            raise RuntimeError("ODE requires 10 initial conditions, {} were given".format(len(self.ode_init_con)))


@attr.s
class ConfigParams:
    """
    Datastructure for the runtime configuration options.
    """
    start = attr.ib()
    stop = attr.ib()
    num_time = attr.ib()
    max_steps = attr.ib()
    relative_tolerance = attr.ib()
    absolute_tolerance = attr.ib()
    enable_taylor_jump = attr.ib()
    taylor_jumps_num = attr.ib()
    taylor_jump_threshold = attr.ib()
    enable_tstop = attr.ib()
    tstop_times = attr.ib(default=attr.Factory(list))


@attr.s
class Solution:
    """
    Datastructure of solutions.
    """
    times = attr.ib()
    solution = attr.ib()
    flags = attr.ib(default=attr.Factory(list))
    tstop_times = attr.ib(default=attr.Factory(list))
    jump_times = attr.ib(default=attr.Factory(list))


@attr.s
class InternalData:
    """
    Datastructure for internally solved params
    """
    params = attr.ib(default=attr.Factory(list))
    derivs = attr.ib(default=attr.Factory(list))
    times = attr.ib(default=attr.Factory(list))
    a1 = attr.ib(default=attr.Factory(list))
    a1dot = attr.ib(default=attr.Factory(list))
    a2 = attr.ib(default=attr.Factory(list))
    a2dot = attr.ib(default=attr.Factory(list))
    a3 = attr.ib(default=attr.Factory(list))
    a3dot = attr.ib(default=attr.Factory(list))
    θ = attr.ib(default=attr.Factory(list))
    θdot = attr.ib(default=attr.Factory(list))
    ϕ = attr.ib(default=attr.Factory(list))
    ϕdot = attr.ib(default=attr.Factory(list))
    problems = attr.ib(default=attr.Factory(list))

    def _finalise(self):
        """
        Finalise data for storage in hdf5 files
        """
        self.derivs = asarray(self.derivs)
        if self.derivs.size == 0:
            self.derivs.shape = (0, len(ODEIndex))

        self.params = asarray(self.params)
        if self.params.size == 0:
            self.params.shape = (0, len(ODEIndex))

        self.times = asarray(self.times)
        self.a1 = asarray(self.a1)
        self.a1dot = asarray(self.a1dot)
        self.a2 = asarray(self.a2)
        self.a2dot = asarray(self.a2dot)
        self.a3 = asarray(self.a3)
        self.a3dot = asarray(self.a3dot)
        self.θ = asarray(self.θ)
        self.θdot = asarray(self.θdot)
        self.ϕdot = asarray(self.ϕdot)

    def __add__(self, other):
        """
        This is required for adding two internal data sets together
        """
        self._finalise()
        other._finalise()

        return InternalData(
            derivs=concatenate((self.derivs, other.derivs)),
            params=concatenate((self.params, other.params)),
            times=concatenate((self.times, other.times)),
            a1=concatenate((self.a1, other.a1)),
            a1dot=concatenate((self.a1dot, other.a1dot)),
            a2=concatenate((self.a1, other.a1)),
            a2dot=concatenate((self.a1dot, other.a1dot)),
            a3=concatenate((self.a1, other.a1)),
            a3dot=concatenate((self.a1dot, other.a1dot)),
            ϕdot=concatenate((self.ϕdot, other.ϕdot)),
            problems=self.problems + other.problems,
        )


@attr.s
class SolutionPhysicalUnits:
    """
    Datastructure of solutions with physical units
    """
    times = attr.ib()
    solution = attr.ib()
    ρ_real = attr.ib()
    ρ_tides = attr.ib()
    ρ_pressure = attr.ib()
    ρ_real_over_ρ_tides = attr.ib()
    ρ_pressure_over_ρ_tides = attr.ib()
    ρ_real_over_ρ_pressure = attr.ib()
    mass = attr.ib()
    len_scale = attr.ib()
    time_scale = attr.ib()
    mass_scale = attr.ib()

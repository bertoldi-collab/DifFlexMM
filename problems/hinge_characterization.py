
from dataclasses import dataclass
import dataclasses
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from jax import grad, jit, value_and_grad, vmap, flatten_util
import tqdm
import nlopt

from difflexmm.dynamics import setup_dynamic_solver
from difflexmm.energy import build_contact_energy, build_strain_energy, combine_block_energies, ligament_energy, ligament_energy_linearized
from difflexmm.geometry import RotatedSquareGeometry
from difflexmm.utils import ContactParams, ControlParams, GeometricalParams, LigamentParams, MechanicalParams, SolutionData, SolutionType
import jax.numpy as jnp


@dataclass
class ForwardProblem:
    """
    Forward problem for hinge characterization using static tests on rotating square samples.
    The forward perform either a tension, compression, or shear test on the sample in displacement control with clamped top and bottom rows.
    """

    # RotatedSquareGeometry
    n1_cells: int
    n2_cells: int
    spacing: Any
    bond_length: Any
    initial_angle: Any

    # Mechanical
    k_stretch: Any
    k_shear: Any
    k_rot: Any
    density: Any
    damping: Any

    # Loading
    loading_type: Literal["tension", "compression", "shear"]
    amplitude: Any
    loading_rate: Any

    # Analysis
    n_timepoints: int
    linearized_strains: bool = False

    # Force multiplier (1 or -1 used to compare with experiments according to assumed sign convention)
    force_multiplier: float = 1.

    # Contact
    use_contact: bool = True
    k_contact: Any = 1.
    min_angle: Any = 0.*jnp.pi/180
    cutoff_angle: Any = 5.*jnp.pi/180

    # Solution or list of solutions
    solution_data: Optional[Union[SolutionType, List[SolutionType]]] = None

    # Solver tolerance
    atol: float = 1e-8
    rtol: float = 1e-8

    # Problem name
    name: str = "hinge_characterization"

    # Flag indicating that solve method is not available. It needs to be set up by calling self.setup().
    is_setup: bool = False

    def setup(self) -> None:
        """
        Set up forward solver.
        """

        # Geometry
        geometry = RotatedSquareGeometry(n1_cells=self.n1_cells,
                                         n2_cells=self.n2_cells,
                                         spacing=self.spacing,
                                         bond_length=self.bond_length)
        block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors = geometry.get_parametrization()
        _block_centroids = block_centroids(self.initial_angle)
        _centroid_node_vectors = centroid_node_vectors(self.initial_angle)
        _bond_connectivity = bond_connectivity()
        _reference_bond_vectors = reference_bond_vectors()

        # Damping
        damped_blocks = jnp.arange(geometry.n_blocks)
        k_ref = self.k_stretch
        mass_ref = self.density*geometry.spacing**2
        damping_ref = jnp.array([
            (k_ref * mass_ref)**0.5,
            (k_ref * mass_ref)**0.5,
            (k_ref * mass_ref)**0.5 * geometry.spacing**2
        ]) * jnp.ones((geometry.n_blocks, 3))
        damping_values = self.damping * damping_ref

        # Applied displacements
        constrained_blocks = jnp.concatenate([
            jnp.arange(geometry.n_blocks-geometry.n1_blocks,
                       geometry.n_blocks),  # top row
            jnp.arange(geometry.n1_blocks)  # bottom row
        ])
        constrained_block_DOF_pairs = jnp.array([
            jnp.concatenate(
                [constrained_blocks, constrained_blocks, constrained_blocks]),
            jnp.concatenate([
                0*jnp.ones(constrained_blocks.shape, dtype=jnp.int32),
                1*jnp.ones(constrained_blocks.shape, dtype=jnp.int32),
                2*jnp.ones(constrained_blocks.shape, dtype=jnp.int32)
            ])
        ]).T
        if self.loading_type == "tension":
            loading_vector = jnp.zeros((len(constrained_block_DOF_pairs),))
            top_row = jnp.where(constrained_block_DOF_pairs[:, 1] == 1)[
                0][:geometry.n1_blocks]  # top positions
            loading_vector = loading_vector.at[top_row].set(1.)
        elif self.loading_type == "compression":
            loading_vector = jnp.zeros((len(constrained_block_DOF_pairs),))
            top_row = jnp.where(constrained_block_DOF_pairs[:, 1] == 1)[
                0][:geometry.n1_blocks]  # top positions
            loading_vector = loading_vector.at[top_row].set(-1.)
        elif self.loading_type == "shear":
            loading_vector = jnp.zeros((len(constrained_block_DOF_pairs),))
            top_row = jnp.where(constrained_block_DOF_pairs[:, 1] == 0)[
                0][:geometry.n1_blocks]  # top positions
            loading_vector = loading_vector.at[top_row].set(1.)
        else:
            raise ValueError(
                "Loading type should be either tension, compression, or shear!"
            )
        # Reaction DOFs to compute reaction forces
        reaction_block_DOF_pairs = constrained_block_DOF_pairs[top_row]

        # Applied displacement
        def applied_displacement(t, amplitude, loading_rate):
            return amplitude * jnp.where(t < loading_rate**-1, t * loading_rate, 1.)

        def constrained_DOFs_fn(t, amplitude, loading_rate):
            # Ramp up to target displacement
            return loading_vector * applied_displacement(t, amplitude, loading_rate)

        # Construct strain energy
        strain_energy = build_strain_energy(
            bond_connectivity=_bond_connectivity,
            bond_energy_fn=ligament_energy_linearized if self.linearized_strains else ligament_energy,
        )
        contact_energy = build_contact_energy(
            bond_connectivity=_bond_connectivity)
        potential_energy = combine_block_energies(
            strain_energy, contact_energy) if self.use_contact else strain_energy

        # Setup solver
        solve_dynamics = setup_dynamic_solver(
            geometry=geometry,
            energy_fn=potential_energy,
            constrained_block_DOF_pairs=constrained_block_DOF_pairs,
            constrained_DOFs_fn=constrained_DOFs_fn,
            damped_blocks=damped_blocks,
            atol=self.atol,
            rtol=self.rtol,
        )

        # Analysis params
        simulation_time = self.loading_rate**-1
        timepoints = jnp.linspace(0, simulation_time, self.n_timepoints)

        # Initial conditions
        state0 = jnp.zeros((2, geometry.n_blocks, 3))

        # Setup forward
        def forward(k_values: Tuple[float, float, float]) -> Tuple[SolutionData, ControlParams]:

            # Design variables
            k_stretch, k_shear, k_rot = k_values

            # Define control params
            control_params = ControlParams(
                geometrical_params=GeometricalParams(
                    block_centroids=_block_centroids,
                    centroid_node_vectors=_centroid_node_vectors,
                ),
                mechanical_params=MechanicalParams(
                    bond_params=LigamentParams(
                        k_stretch=k_stretch,
                        k_shear=k_shear,
                        k_rot=k_rot,
                        reference_vector=_reference_bond_vectors,
                    ),
                    density=self.density,
                    damping=damping_values,
                    contact_params=ContactParams(
                        k_contact=self.k_contact,
                        min_angle=self.min_angle,
                        cutoff_angle=self.cutoff_angle,
                    ),
                ),
                constraint_params=dict(
                    amplitude=self.amplitude,
                    loading_rate=self.loading_rate,
                ),
            )

            # Solve dynamics
            solution = solve_dynamics(
                state0=state0,
                timepoints=timepoints,
                control_params=control_params,
            )

            return SolutionData(
                block_centroids=_block_centroids,
                centroid_node_vectors=_centroid_node_vectors,
                bond_connectivity=_bond_connectivity,
                timepoints=timepoints,
                fields=solution
            ), control_params

        self.solve = jit(forward)
        self.geometry = geometry
        self.potential_energy = potential_energy
        self.elastic_forces = grad(potential_energy)
        self.applied_displacement = applied_displacement
        self.reaction_block_DOF_pairs = reaction_block_DOF_pairs
        self.is_setup = True

    def force_displacement(self, solution_data: SolutionData, control_params: ControlParams):
        """
        Compute force-displacement data for the given solution data and control params.
        """
        if self.is_setup:
            displacement_history = solution_data.fields[:, 0]
            block_DOF_pairs = self.reaction_block_DOF_pairs
            force_history = vmap(
                lambda u: jnp.sum(
                    self.elastic_forces(u, control_params)[
                        block_DOF_pairs[:, 0],
                        block_DOF_pairs[:, 1],
                    ]
                )
            )(displacement_history)
            applied_u = self.applied_displacement(
                solution_data.timepoints,
                **control_params.constraint_params,
            )
            return jnp.array([applied_u, force_history*self.force_multiplier])

    @staticmethod
    def from_data(problem_data):
        problem_data = ForwardProblem(**problem_data)
        problem_data.is_setup = False
        return problem_data

    def to_data(self):
        return ForwardProblem(**dataclasses.asdict(self))

    @staticmethod
    def from_dict(dict_in):
        # Convert solution data to named tuple
        if dict_in["solution_data"] is not None:
            if type(dict_in["solution_data"]) is dict:
                dict_in["solution_data"] = SolutionData(
                    **dict_in["solution_data"])
            elif type(dict_in["solution_data"]) is list:
                dict_in["solution_data"] = [SolutionData(
                    **solution) for solution in dict_in["solution_data"]]
        problem_data = ForwardProblem(**dict_in)
        problem_data.is_setup = False
        return problem_data

    def to_dict(self):
        # Make sure namedtuples are converted to dictionaries before saving
        dict_out = dataclasses.asdict(self)
        if type(dict_out["solution_data"]) is SolutionData:
            dict_out["solution_data"] = dict_out["solution_data"]._asdict()
        elif type(dict_out["solution_data"]) is list:
            dict_out["solution_data"] = [solution._asdict()
                                         for solution in dict_out["solution_data"]]
        return dict_out


def resample(x, y, n_timepoints):
    return jnp.interp(
        jnp.linspace(jnp.min(x), jnp.max(x), n_timepoints),
        x,  # displacement history
        y  # force history
    )


def enforce_bounds(x, lower=None, upper=None):
    if lower is not None:
        x = jnp.where(x <= lower, lower, x)
    if upper is not None:
        x = jnp.where(x >= upper, upper, x)

    return x


def naive_GD(objective_fn, initial_guess, step_size, n_iterations, lower_bound=None, upper_bound=None):
    """
    Naive gradient descent optimization with fixed step size and bounds.
    """

    # Compile objective and its gradient
    obj_and_grad = jit(value_and_grad(objective_fn))
    obj_values = []
    design_values = [initial_guess]  # initial design

    with tqdm.trange(n_iterations) as pbar:
        for i in pbar:
            value, update = obj_and_grad(design_values[i])
            obj_values += [value]
            new_design = tuple(
                enforce_bounds(x_i - step_size * dx,
                               lower=lower_bound, upper=upper_bound)
                for x_i, dx in zip(design_values[i], update)
            )
            design_values += [new_design]
            pbar.set_description(f"Objective = {value:.9f}")

    return obj_values, design_values


@dataclass
class OptimizationProblem:
    """
    Optimization problem for hinge characterization using static tests on rotating square samples.
    """

    forward_problems: List[ForwardProblem]
    # NOTE: Each response is an array of the form [displacement_history, force_history, force_std].
    target_responses: Dict[str, jnp.ndarray]
    fitted_responses: Optional[Dict[str, jnp.ndarray]] = None
    objective_values: Optional[List[Any]] = None
    design_values: Optional[List[Any]] = None
    name: str = ForwardProblem.name

    # Flag indicating that objective_fn method is not available. It needs to be set up by calling self.setup_objective().
    is_setup: bool = False

    def __post_init__(self):
        self.objective_values = [] if self.objective_values is None else self.objective_values
        self.design_values = [] if self.design_values is None else self.design_values

    def compute_fitted_responses(self, k_values: Tuple[float, float, float]):
        """
        Compute the fitted force-displacement data for the given k_values for all the forwards.
        """

        # Make sure forward solvers are set up
        for problem in self.forward_problems:
            if not problem.is_setup:
                problem.setup()

        return {problem.loading_type: problem.force_displacement(*problem.solve(k_values)) for problem in self.forward_problems}

    def setup_objective(self) -> None:
        """
        Jit compiles the objective function.
        """

        # Make sure forward solvers are set up
        for problem in self.forward_problems:
            if not problem.is_setup:
                problem.setup()

        # Ensure targets and simulations are sampled in the same way
        # NOTE: This assumes that the displacement history is a linear ramp!
        n_timepoints = self.forward_problems[0].n_timepoints
        target_forces = jnp.array([resample(target_u, target_f, n_timepoints)
                                   for target_u, target_f, _ in self.target_responses.values()])

        def response_squared_error(k_values: Tuple[float, float, float]):

            fitted_responses = self.compute_fitted_responses(k_values)
            reaction_forces = jnp.array([
                forces for _, forces in fitted_responses.values()
            ])

            # Compute distance between target and fitted forces
            return jnp.mean((reaction_forces - target_forces)**2)

        self.objective_fn = response_squared_error
        self.is_setup = True

    def run_optimization_GD(self, initial_guess, n_iterations: int, step_size: float, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None):

        # Make sure objective_fn is set up
        if not self.is_setup:
            self.setup_objective()

        self.objective_values, self.design_values = naive_GD(
            objective_fn=self.objective_fn,
            initial_guess=initial_guess,
            step_size=step_size,
            n_iterations=n_iterations,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

        self.fitted_responses = self.compute_fitted_responses(
            self.design_values[-1])

    def run_optimization_nlopt(
            self,
            # Tuple[float, float, float] initial values for k_stretch, k_shear, k_rot
            initial_guess,
            n_iterations: int,
            max_time: Optional[int] = None,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None,):

        # Make sure objective_fn is set up
        if not self.is_setup:
            self.setup_objective()

        def flatten(tree): return flatten_util.ravel_pytree(tree)[0]
        _, unflatten = flatten_util.ravel_pytree(initial_guess)

        objective_and_grad = jit(value_and_grad(self.objective_fn))

        def nlopt_objective(x, grad):

            v, g = objective_and_grad(unflatten(x))  # jax evaluation
            self.objective_values.append(v)
            self.design_values.append(unflatten(x))

            print(
                f"Iteration: {len(self.objective_values)}\nObjective = {self.objective_values[-1]}")

            if grad.size > 0:
                grad[:] = flatten(g)

            return float(v)

        initial_guess_flattened = flatten(initial_guess)
        opt = nlopt.opt(nlopt.LD_MMA, len(initial_guess_flattened))

        opt.set_param("verbosity", 1)
        opt.set_maxeval(n_iterations)

        opt.set_min_objective(nlopt_objective)

        if lower_bound is not None:
            opt.set_lower_bounds(lower_bound)
        if upper_bound is not None:
            opt.set_upper_bounds(upper_bound)

        if max_time is not None:
            opt.set_maxtime(max_time)

        # Run optimization
        opt.optimize(initial_guess_flattened)

        self.fitted_responses = self.compute_fitted_responses(
            self.design_values[-1])

    @staticmethod
    def from_data(optimization_data):
        optimization_data.forward_problems = [
            ForwardProblem.from_data(problem)
            for problem in optimization_data.forward_problems
        ]
        optimization_data.is_setup = False
        return optimization_data

    def to_data(self):
        return OptimizationProblem(**dataclasses.asdict(self))

    @staticmethod
    def from_dict(dict_in):
        # Convert solution data to named tuple
        dict_in["forward_problems"] = [
            ForwardProblem.from_dict(problem_data)
            for problem_data in dict_in["forward_problems"]
        ]
        optimization_data = OptimizationProblem(**dict_in)
        optimization_data.is_setup = False
        return optimization_data

    def to_dict(self):
        # Make sure namedtuples are converted to dictionaries before saving
        dict_out = dataclasses.asdict(self)
        dict_out["forward_problems"] = [problem.to_dict()
                                        for problem in self.forward_problems]
        return dict_out

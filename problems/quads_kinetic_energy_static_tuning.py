from difflexmm.utils import SolutionType, SolutionData, ControlParams, GeometricalParams, MechanicalParams, LigamentParams, ContactParams
from difflexmm.geometry import QuadGeometry, compute_inertia, compute_edge_angles, compute_edge_lengths
from difflexmm.energy import build_strain_energy, kinetic_energy, ligament_energy, ligament_energy_linearized, build_contact_energy, combine_block_energies, compute_ligament_strains_history
from difflexmm.dynamics import setup_dynamic_solver
from typing import Any, Optional, List, Union, Tuple, Dict
import dataclasses
from dataclasses import dataclass

import nlopt
import jax.numpy as jnp
from jax import flatten_util, pmap
from jax import value_and_grad, jit, vmap, jacobian


@dataclass
class ForwardInput:
    """
    Input params for the forward solve function.
    As we will be using pmap, each loading param needs to be a tuple of length equal.

    Args:
        horizontal_shifts (ndarray): initial guess horizontal shifts.
        vertical_shifts (ndarray): initial guess vertical shifts.
        amplitude (Tuple[Any, ...]): amplitude of the dynamic loading.
        loading_rate (Tuple[Any, ...]): loading rate of the dynamic loading.
        compressive_strain (Tuple[Any, ...]): compressive strain of the static loading.
        compressive_strain_rate (Tuple[Any, ...]): compressive strain rate of the static loading.
    """

    # Geometry
    horizontal_shifts: Any  # initial guess horizontal shifts
    vertical_shifts: Any  # initial guess vertical shifts

    # Dynamic loading
    amplitude: Tuple[Any, ...]
    loading_rate: Tuple[Any, ...]

    # Static loading
    compressive_strain: Tuple[Any, ...]
    compressive_strain_rate: Tuple[Any, ...]


@dataclass
class ForwardProblem:
    """
    Forward problem class multi-tasking via static compression.
    BCs:
        - Clamped bottom and top edges.
        - Single dynamic input on the left edge.
    """

    # QuadGeometry
    n1_blocks: int
    n2_blocks: int
    spacing: Any
    bond_length: Any

    # Mechanical
    k_stretch: Any
    k_shear: Any
    k_rot: Any
    density: Any
    damping: Any

    # Dynamic loading
    # amplitude: Any
    # loading_rate: Any
    n_excited_blocks: int
    input_shift: int

    # Static loading
    # compressive_strain: Any
    # compressive_strain_rate: Any

    # Analysis params
    # time for the dynamic loading: total time = simulation_time + compressive_strain*compressive_strain_rate**-1 + input_delay
    simulation_time_dynamic: Any
    n_timepoints: int
    linearized_strains: bool = False

    # Contact
    use_contact: bool = True
    k_contact: Any = 1.
    min_angle: Any = 0.*jnp.pi/180
    cutoff_angle: Any = 5.*jnp.pi/180

    # Solution or list of solutions
    solution_data: Optional[Union[SolutionType, List[SolutionType]]] = None

    # Problem name
    name: str = "quads_kinetic_energy_static_tuning"

    # Solver tolerance
    atol: float = 1e-8
    rtol: float = 1e-8

    # Flag indicating that solve method is not available. It needs to be set up by calling self.setup().
    is_setup: bool = False

    def setup(self, excited_blocks_fn=None) -> None:
        """
        Set up forward solver.
        """

        # Geometry
        geometry = QuadGeometry(
            n1_blocks=self.n1_blocks,
            n2_blocks=self.n2_blocks,
            spacing=self.spacing,
            bond_length=self.bond_length
        )
        block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors = geometry.get_parametrization()
        # Compute bond connectivity once as it is constant
        _bond_connectivity = bond_connectivity()
        # Compute reference bond vectors once as they are constant
        _reference_bond_vectors = reference_bond_vectors()

        # Initial conditions
        state0 = jnp.zeros((2, geometry.n_blocks, 3))

        # Damping
        damped_blocks = jnp.arange(geometry.n_blocks)

        # Dynamic input and BCs
        n_excited_blocks = self.n_excited_blocks
        input_shift = self.input_shift  # Vertical shift for the applied loading
        driven_block_DOF_pairs = jnp.array([
            jnp.tile(
                jnp.arange((geometry.n2_blocks-n_excited_blocks)//2 + input_shift,
                           (geometry.n2_blocks+n_excited_blocks)//2 + input_shift) * geometry.n1_blocks,
                3),
            jnp.array([0]*n_excited_blocks + [1] *
                      n_excited_blocks + [2]*n_excited_blocks)
        ]).T
        # Clamped top and bottom sides
        clamped_block_DOF_pairs_bottom = jnp.array([
            jnp.concatenate([jnp.arange(0, geometry.n1_blocks)]*3),
            jnp.array([1]*geometry.n1_blocks + [0] *
                      geometry.n1_blocks + [2]*geometry.n1_blocks)
        ]).T
        clamped_block_DOF_pairs_top = jnp.array([
            jnp.concatenate(
                [jnp.arange(geometry.n_blocks-geometry.n1_blocks, geometry.n_blocks)]*3),
            jnp.array([1]*geometry.n1_blocks + [0] *
                      geometry.n1_blocks + [2]*geometry.n1_blocks)
        ]).T
        constrained_block_DOF_pairs = jnp.concatenate(
            [driven_block_DOF_pairs, clamped_block_DOF_pairs_bottom,
                clamped_block_DOF_pairs_top]
        )

        constrained_DOFs_loading_vector = jnp.zeros(
            (len(constrained_block_DOF_pairs),))
        constrained_DOFs_loading_vector_dynamic = constrained_DOFs_loading_vector.at[:n_excited_blocks].set(
            1)  # Dynamic loading
        constrained_DOFs_loading_vector_static = constrained_DOFs_loading_vector.at[3*n_excited_blocks:3*n_excited_blocks+geometry.n1_blocks].set(
            0.5)  # Compression bottom
        constrained_DOFs_loading_vector_static = constrained_DOFs_loading_vector_static.at[3*n_excited_blocks+3*geometry.n1_blocks:3*n_excited_blocks+4*geometry.n1_blocks].set(
            -0.5)  # Compression top

        clamped_blocks_ids = jnp.unique(
            jnp.concatenate([
                clamped_block_DOF_pairs_bottom, clamped_block_DOF_pairs_top,
            ])[:, 0]
        )
        moving_blocks_ids = jnp.setdiff1d(
            jnp.arange(geometry.n_blocks),
            clamped_blocks_ids
        )
        driven_blocks_ids = jnp.unique(driven_block_DOF_pairs[:, 0])

        # Dynamic loading
        if excited_blocks_fn is None:
            # Apply sinthetic pulse loading
            # NOTE: This is used for optimization.
            def constrained_DOFs_fn_dynamic(t, amplitude, loading_rate):
                return amplitude * jnp.where(
                    (t > 0.) & (t < loading_rate**-1),
                    (1 - jnp.cos(2*jnp.pi * loading_rate * t))/2,
                    0.
                ) * constrained_DOFs_loading_vector_dynamic
        else:
            def constrained_DOFs_fn_dynamic(t, *args, **kwargs):
                return excited_blocks_fn(t) * constrained_DOFs_loading_vector_dynamic

        # Static loading

        def constrained_DOFs_fn_static(t, compressive_strain, compressive_strain_rate):
            return (geometry.n2_blocks-1)*geometry.spacing * jnp.where(
                t < compressive_strain*compressive_strain_rate**-1,
                t * compressive_strain_rate,
                compressive_strain
            ) * constrained_DOFs_loading_vector_static

        def constrained_DOFs_fn(t, amplitude, loading_rate, compressive_strain, compressive_strain_rate, input_delay):
            return constrained_DOFs_fn_static(t, compressive_strain, compressive_strain_rate) + constrained_DOFs_fn_dynamic(t - compressive_strain*compressive_strain_rate**-1 - input_delay, amplitude, loading_rate)

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

        # Setup forward
        def forward(
                # Design variables
                horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray],
                # Other forward inputs
                amplitude: float,
                loading_rate: float,
                compressive_strain: float,
                compressive_strain_rate: float,
                full_simulation_time: bool = False,
                n_timepoints: int = self.n_timepoints,) -> SolutionData:

            # Design variables
            horizontal_shifts, vertical_shifts = horizontal_vertical_shifts
            input_delay = 0.1 * loading_rate**-1

            # Define control params
            control_params = ControlParams(
                geometrical_params=GeometricalParams(
                    block_centroids=block_centroids(
                        horizontal_shifts, vertical_shifts),
                    centroid_node_vectors=centroid_node_vectors(
                        horizontal_shifts, vertical_shifts),
                ),
                mechanical_params=MechanicalParams(
                    bond_params=LigamentParams(
                        k_stretch=self.k_stretch,
                        k_shear=self.k_shear,
                        k_rot=self.k_rot,
                        reference_vector=_reference_bond_vectors,
                    ),
                    density=self.density,
                    damping=self.damping,
                    contact_params=ContactParams(
                        k_contact=self.k_contact,
                        min_angle=self.min_angle,
                        cutoff_angle=self.cutoff_angle,
                    ),
                ),
                constraint_params=dict(
                    amplitude=amplitude,
                    loading_rate=loading_rate,
                    compressive_strain=compressive_strain,
                    compressive_strain_rate=compressive_strain_rate,
                    input_delay=input_delay,
                ),
            )

            # Analysis params
            if full_simulation_time:
                # Used to compute the solution including the static step
                simulation_time = self.simulation_time_dynamic + \
                    compressive_strain*compressive_strain_rate**-1 + input_delay
                timepoints = jnp.linspace(0, simulation_time, n_timepoints)
            else:
                # Used to compute the solution on dynamic step only (used for optimization)
                timepoints = jnp.concatenate([
                    # Add initial timepoint to ensure correct initial conditions
                    jnp.array([0.]),
                    jnp.linspace(compressive_strain*compressive_strain_rate**-1 +
                                 input_delay, compressive_strain*compressive_strain_rate**-1 +
                                 input_delay + self.simulation_time_dynamic, n_timepoints)
                ])

            # Solve dynamics
            solution = solve_dynamics(
                state0=state0,
                timepoints=timepoints,
                control_params=control_params,
            )

            return SolutionData(
                block_centroids=block_centroids(
                    horizontal_shifts, vertical_shifts),
                centroid_node_vectors=centroid_node_vectors(
                    horizontal_shifts, vertical_shifts),
                bond_connectivity=_bond_connectivity,
                timepoints=timepoints if full_simulation_time else timepoints[1:]-timepoints[1],
                fields=solution if full_simulation_time else solution[1:],
            )

        # Used for optimization
        self.solve_dynamic = lambda *args, **kwargs: forward(*args, **kwargs,
                                                             full_simulation_time=False, n_timepoints=self.n_timepoints)
        self.solve = forward
        self.geometry = geometry
        self.clamped_blocks_ids = clamped_blocks_ids
        self.moving_blocks_ids = moving_blocks_ids
        self.driven_blocks_ids = driven_blocks_ids
        self.is_setup = True

    def compute_response_data(self, solution_data: Optional[SolutionData] = None) -> dict:
        """Compute the response data associated with the solution data.

        This function computes the response data, such as the strains and the strain energy. If the solution data is not
        provided, the solution data stored in the class is used.
        The fields of the returned dictionary are:
            - all the fields of the SolutionData namedtuple
            - strain_energy_stretch: the strain energy due to stretching format as (n_timepoints, n_bonds).
            - strain_energy_shear: the strain energy due to shearing format as (n_timepoints, n_bonds).
            - strain_energy_bending: the strain energy due to bending format as (n_timepoints, n_bonds).
            - kinetic_energy: the kinetic energy format as (n_timepoints, n_blocks).

        Raises:
            ValueError: If the solution data is not available.
            ValueError: If the solution data is not of type SolutionData.

        Returns:
            dict: Dictionary with the response data.
        """

        if not self.is_setup:
            self.setup()

        if solution_data is None:
            if self.solution_data is None:
                raise ValueError("No solution data available!")
            else:
                solution_data = self.solution_data

        if type(solution_data) is not SolutionData:
            raise ValueError("Solution data is not of type SolutionData!")

        # Return a dictionary with the solution info
        dict_out = solution_data._asdict()
        # Compute strains
        axial_strain, shear_strain, bending_strain = compute_ligament_strains_history(
            solution_data.fields[:, 0],
            solution_data.centroid_node_vectors,
            solution_data.bond_connectivity,
            self.geometry.reference_bond_vectors()
        )
        dict_out["strain_energy_stretch"] = 0.5 * \
            self.k_stretch * (axial_strain*self.bond_length)**2
        dict_out["strain_energy_shear"] = 0.5 * \
            self.k_shear * (shear_strain*self.bond_length)**2
        dict_out["strain_energy_bending"] = 0.5 * \
            self.k_rot * bending_strain**2
        # Add kinetic energy to the dictionary
        inertia = compute_inertia(
            solution_data.centroid_node_vectors, self.density)
        dict_out["kinetic_energy"] = jnp.sum(
            0.5 * solution_data.fields[:, 1]**2 * inertia, axis=-1)

        return dict_out

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


@dataclass
class OptimizationProblem:
    """
    Optimization problem class for kinetic-energy-based multi-tasking via static compression.
    Objective is a weighted combination of the kinetic energy of each forward problem. Use negative weights to minimize (aka "protect").
    ForwardInput is used to define the set of forward problems.
    """

    # Forward problem provides a forward function that can be called on design variables and compressive strain
    forward_problem: ForwardProblem
    forward_input: ForwardInput
    target_sizes: Tuple[Tuple[int, int], ...]
    target_shifts: Tuple[Tuple[int, int], ...]
    weights: Tuple[float, ...]
    objective_values: Optional[List[Any]] = None
    objective_values_individual: Optional[List[Any]] = None
    design_values: Optional[List[Any]] = None
    constraints_violation: Optional[Dict[str, List[Any]]] = None
    name: str = ForwardProblem.name

    # Flag indicating that objective_fn method is not available. It needs to be set up by calling self.setup_objective().
    is_setup: bool = False

    def __post_init__(self):
        self.objective_values = [] if self.objective_values is None else self.objective_values
        self.objective_values_individual = [
        ] if self.objective_values_individual is None else self.objective_values_individual
        self.design_values = [] if self.design_values is None else self.design_values
        self.constraints_violation = {
            "angles": [], "edge_lengths": []
        } if self.constraints_violation is None else self.constraints_violation

    def setup_objective(self) -> None:
        """
        Jit compiles the objective function.
        """

        # Make sure forward solvers are set up
        if not self.forward_problem.is_setup:
            self.forward_problem.setup()

        # Retrieve geometry and density
        geometry = self.forward_problem.geometry
        density = self.forward_problem.density

        # Target blocks at shifted location from the middle of the domain
        target_blocks_array = jnp.array([
            jnp.array([
                j * geometry.n1_blocks + i
                for i in range((geometry.n1_blocks-target_size[0])//2 + target_shift[0], (geometry.n1_blocks+target_size[0])//2 + target_shift[0])
                for j in range((geometry.n2_blocks-target_size[1])//2 + target_shift[1], (geometry.n2_blocks+target_size[1])//2 + target_shift[1])
            ])
            for target_size, target_shift in zip(self.target_sizes, self.target_shifts)
        ])

        forward_input_array = jnp.array([
            self.forward_input.amplitude,
            self.forward_input.loading_rate,
            self.forward_input.compressive_strain,
            self.forward_input.compressive_strain_rate,
        ]).T

        # Energy for a single target
        def target_kinetic_energy(horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray], forward_input: Tuple[float, ...], target_blocks: jnp.ndarray):
            solution_data = self.forward_problem.solve_dynamic(
                horizontal_vertical_shifts, *forward_input)
            return kinetic_energy(
                block_velocity=solution_data.fields[:, 1, target_blocks, :],
                inertia=compute_inertia(
                    vertices=solution_data.centroid_node_vectors,
                    density=density
                )[target_blocks]
            )

        target_kinetic_energy_mapped = pmap(
            target_kinetic_energy, in_axes=(None, 0, 0))

        # Total objective is the weighted sum of the kinetic energy for each forward
        def total_objective(horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray]):
            return jnp.array(self.weights) @ target_kinetic_energy_mapped(horizontal_vertical_shifts, forward_input_array, target_blocks_array)

        self.objective_fn = total_objective
        self.objective_fn_individual = lambda horizontal_vertical_shifts: target_kinetic_energy_mapped(
            horizontal_vertical_shifts, forward_input_array, target_blocks_array)
        self.is_setup = True
        self.target_blocks = target_blocks_array

    def setup_angle_constraints(self, min_void_angle=0., min_block_angle=0.):
        centroid_node_vectors = self.forward_problem.geometry.centroid_node_vectors
        bond_connectivity = self.forward_problem.geometry.bond_connectivity()

        def angle_constraints(horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray]):

            node_vectors = centroid_node_vectors(*horizontal_vertical_shifts)
            void_angles_1, void_angles_2, block_angles_1, block_angles_2 = vmap(
                lambda b: jnp.mod(
                    jnp.array(compute_edge_angles(node_vectors, b)), 2*jnp.pi),
                in_axes=0
            )(bond_connectivity).T

            return jnp.concatenate([
                -(void_angles_1 - min_void_angle),
                -(void_angles_2 - min_void_angle),
                -(block_angles_1 - min_block_angle),
                -(block_angles_2 - min_block_angle),
            ])

        self.angle_constraints = angle_constraints

    def setup_edge_length_constraints(self, min_edge_length):
        centroid_node_vectors = self.forward_problem.geometry.centroid_node_vectors

        def edge_length_constraints(horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray]):
            edge_lengths = compute_edge_lengths(
                centroid_node_vectors(*horizontal_vertical_shifts)).reshape(-1)
            return -(edge_lengths - min_edge_length)

        self.edge_length_constraints = edge_length_constraints

    def run_optimization_nlopt(
            self,
            initial_guess,
            n_iterations: int,
            max_time: Optional[int] = None,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None,
            min_void_angle: Optional[float] = None,
            min_block_angle: Optional[float] = None,
            min_edge_length: Optional[float] = None,):

        # Make sure objective_fn is set up
        if not self.is_setup:
            self.setup_objective()

        def flatten(tree): return flatten_util.ravel_pytree(tree)[0]
        _, unflatten = flatten_util.ravel_pytree(initial_guess)

        # objective_and_grad = jit(value_and_grad(self.objective_fn))
        objective_and_grad = value_and_grad(self.objective_fn)

        def nlopt_objective(x, grad):

            v, g = objective_and_grad(unflatten(x))  # jax evaluation
            vs = self.objective_fn_individual(unflatten(x))
            self.objective_values.append(v)
            self.objective_values_individual.append(vs)
            self.design_values.append(unflatten(x))

            print(
                f"Iteration: {len(self.objective_values)}\nObjective = {self.objective_values[-1]}\nObjectives = {self.objective_values_individual[-1]}"
            )

            if grad.size > 0:
                grad[:] = flatten(g)

            return float(v)

        initial_guess_flattened = flatten(initial_guess)
        opt = nlopt.opt(nlopt.LD_MMA, len(initial_guess_flattened))

        if min_void_angle is not None and min_block_angle is not None:
            self.setup_angle_constraints(min_void_angle, min_block_angle)
            angle_constraints = jit(
                lambda x: self.angle_constraints(unflatten(x)))
            angle_constraints_jac = jit(
                jacobian(lambda x: self.angle_constraints(unflatten(x))))

            def nlopt_angle_constraints(result, x, grad):

                result[:] = angle_constraints(x)
                self.constraints_violation["angles"].append(result.max())

                print(
                    f"Angle constraints violation = {self.constraints_violation['angles'][-1]}")

                if grad.size > 0:
                    grad[:, :] = angle_constraints_jac(x)

            opt.add_inequality_mconstraint(
                nlopt_angle_constraints,
                1.e-8 *
                jnp.ones(
                    (4*len(self.forward_problem.geometry.bond_connectivity()),))
            )

        if min_edge_length is not None:
            self.setup_edge_length_constraints(min_edge_length)
            edge_length_constraints = jit(
                lambda x: self.edge_length_constraints(unflatten(x)))
            edge_length_constraints_jac = jit(
                jacobian(lambda x: self.edge_length_constraints(unflatten(x))))

            def nlopt_edge_length_constraints(result, x, grad):

                result[:] = edge_length_constraints(x)
                self.constraints_violation["edge_lengths"].append(result.max())

                print(
                    f"Edge length constraints violation = {self.constraints_violation['edge_lengths'][-1]}")

                if grad.size > 0:
                    grad[:, :] = edge_length_constraints_jac(x)

            opt.add_inequality_mconstraint(
                nlopt_edge_length_constraints,
                1.e-8 * jnp.ones(self.forward_problem.geometry.n_blocks *
                                 self.forward_problem.geometry.n_npb)
            )

        opt.set_param("verbosity", 1)
        opt.set_maxeval(n_iterations)

        opt.set_max_objective(nlopt_objective)

        if lower_bound is not None:
            opt.set_lower_bounds(lower_bound)
        if upper_bound is not None:
            opt.set_upper_bounds(upper_bound)

        if max_time is not None:
            opt.set_maxtime(max_time)

        # Run optimization
        opt.optimize(initial_guess_flattened)

        # Store forward solution data for the last design
        self.compute_best_forwards()

    def compute_best_forwards(self, n_timepoints: int = 200):

        if len(self.design_values) == 0:
            raise ValueError("No design has been optimized yet.")

        forward_input_array = jnp.array([
            self.forward_input.amplitude,
            self.forward_input.loading_rate,
            self.forward_input.compressive_strain,
            self.forward_input.compressive_strain_rate,
        ]).T

        if not self.forward_problem.is_setup:
            self.forward_problem.setup()

        self.forward_problem.solution_data = [
            self.forward_problem.solve(
                self.design_values[-1],
                *forward_input,
                full_simulation_time=False,
                n_timepoints=n_timepoints
            )
            for forward_input in forward_input_array
        ]

        return self.forward_problem.solution_data

    @staticmethod
    def from_data(optimization_data):
        optimization_data.forward_problem = ForwardProblem.from_data(
            optimization_data.forward_problem)
        optimization_data.forward_input = ForwardInput(
            **optimization_data.forward_input)
        optimization_data.is_setup = False
        return optimization_data

    def to_data(self):
        return OptimizationProblem(**dataclasses.asdict(self))

    @staticmethod
    def from_dict(dict_in):
        # Convert solution data to named tuple
        dict_in["forward_problem"] = ForwardProblem.from_dict(
            dict_in["forward_problem"])
        dict_in["forward_input"] = ForwardInput(**dict_in["forward_input"])
        optimization_data = OptimizationProblem(**dict_in)
        optimization_data.is_setup = False
        return optimization_data

    def to_dict(self):
        # Make sure namedtuples are converted to dictionaries before saving
        dict_out = dataclasses.asdict(self)
        dict_out["forward_problem"] = self.forward_problem.to_dict()
        return dict_out

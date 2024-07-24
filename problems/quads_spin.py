from difflexmm.utils import SolutionType, SolutionData, ControlParams, GeometricalParams, MechanicalParams, LigamentParams, ContactParams
from difflexmm.geometry import QuadGeometry, compute_inertia, compute_edge_angles, compute_edge_lengths
from difflexmm.energy import build_strain_energy, angular_momentum, ligament_energy, ligament_energy_linearized, build_contact_energy, combine_block_energies
from difflexmm.dynamics import setup_dynamic_solver
from typing import Any, Literal, Optional, List, Union, Tuple, Dict
import dataclasses
from dataclasses import dataclass

import nlopt
import jax.numpy as jnp
from jax import flatten_util
from jax import value_and_grad, jit, vmap, jacobian


@dataclass
class ForwardInput:
    """Input params for the forward solve function."""

    # Geometry
    horizontal_shifts: Any  # initial guess horizontal shifts
    vertical_shifts: Any  # initial guess vertical shifts


@dataclass
class ForwardProblem:
    """
    Forward problem for the quads focusing problem.
    BCs:
        - Clamped corners.
        - Single dynamic input on the specified edge.
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
    amplitude: Any
    loading_rate: Any
    input_delay: Any
    n_excited_blocks: int
    loaded_side: Literal["left", "right", "bottom", "top"]
    input_shift: int

    # Analysis params
    simulation_time: Any
    n_timepoints: int
    linearized_strains: bool = False

    # Contact
    use_contact: bool = True
    k_contact: Any = 1.
    min_angle: Any = 0.*jnp.pi/180
    cutoff_angle: Any = 5.*jnp.pi/180

    # Number of blocks clamped at the corners
    n_blocks_clamped_corner = 2

    # Solution or list of solutions
    solution_data: Optional[Union[SolutionType, List[SolutionType]]] = None

    # Solver tolerance
    atol: float = 1e-8
    rtol: float = 1e-8

    # Flag indicating that solve method is not available. It needs to be set up by calling self.setup().
    is_setup: bool = False

    # Problem name
    name: str = "quads_spin"

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

        # Damping
        damped_blocks = jnp.arange(geometry.n_blocks)

        # Dynamic input and BCs
        n_excited_blocks = self.n_excited_blocks
        input_shift = self.input_shift  # Vertical shift for the applied loading
        if self.loaded_side == "left":
            driven_block_DOF_pairs = jnp.array([
                jnp.tile(
                    jnp.arange((geometry.n2_blocks-n_excited_blocks)//2 + input_shift,
                               (geometry.n2_blocks+n_excited_blocks)//2 + input_shift) * geometry.n1_blocks,
                    3),
                jnp.array([0]*n_excited_blocks + [1] *
                          n_excited_blocks + [2]*n_excited_blocks)
            ]).T
        elif self.loaded_side == "right":
            driven_block_DOF_pairs = jnp.array([
                jnp.tile(
                    jnp.arange((geometry.n2_blocks-n_excited_blocks)//2 + input_shift, (geometry.n2_blocks +
                               n_excited_blocks)//2 + input_shift) * geometry.n1_blocks + (geometry.n1_blocks-1),
                    3),
                jnp.array([0]*n_excited_blocks + [1] *
                          n_excited_blocks + [2]*n_excited_blocks)
            ]).T
        elif self.loaded_side == "bottom":
            driven_block_DOF_pairs = jnp.array([
                jnp.tile(
                    jnp.arange((geometry.n1_blocks-n_excited_blocks)//2 + input_shift,
                               (geometry.n1_blocks+n_excited_blocks)//2 + input_shift),
                    3),
                jnp.array([1]*n_excited_blocks + [0] *
                          n_excited_blocks + [2]*n_excited_blocks)
            ]).T
        elif self.loaded_side == "top":
            driven_block_DOF_pairs = jnp.array([
                jnp.tile(
                    jnp.arange((geometry.n1_blocks-n_excited_blocks)//2 + input_shift,
                               (geometry.n1_blocks+n_excited_blocks)//2 + input_shift) + geometry.n1_blocks*(geometry.n2_blocks-1),
                    3),
                jnp.array([1]*n_excited_blocks + [0] *
                          n_excited_blocks + [2]*n_excited_blocks)
            ]).T
        else:
            raise ValueError(
                f"Unknown loaded_side: {self.loaded_side}. Should be either 'left', 'right', 'bottom' or 'top'."
            )
        # Clamped corners
        clamped_block_DOF_pairs_bl = jnp.array([
            jnp.tile(
                jnp.concatenate([
                    jnp.arange(0, self.n_blocks_clamped_corner),
                    jnp.array(
                        [0+i*geometry.n1_blocks for i in range(1, self.n_blocks_clamped_corner)])
                ]), 3),
            jnp.array([0]*(2*self.n_blocks_clamped_corner-1) + [1] *
                      (2*self.n_blocks_clamped_corner-1) + [2]*(2*self.n_blocks_clamped_corner-1))
        ]).T
        clamped_block_DOF_pairs_br = jnp.array([
            jnp.tile(
                jnp.concatenate([
                    jnp.arange(geometry.n1_blocks -
                               self.n_blocks_clamped_corner, geometry.n1_blocks),
                    jnp.array(
                        [(i+1)*geometry.n1_blocks-1 for i in range(1, self.n_blocks_clamped_corner)])
                ]), 3),
            jnp.array([0]*(2*self.n_blocks_clamped_corner-1) + [1] *
                      (2*self.n_blocks_clamped_corner-1) + [2]*(2*self.n_blocks_clamped_corner-1))
        ]).T
        clamped_block_DOF_pairs_tr = jnp.array([
            jnp.tile(
                jnp.concatenate([
                    jnp.arange(geometry.n_blocks -
                               self.n_blocks_clamped_corner, geometry.n_blocks),
                    jnp.array([geometry.n_blocks-i*geometry.n1_blocks -
                              1 for i in range(1, self.n_blocks_clamped_corner)])
                ]), 3),
            jnp.array([0]*(2*self.n_blocks_clamped_corner-1) + [1] *
                      (2*self.n_blocks_clamped_corner-1) + [2]*(2*self.n_blocks_clamped_corner-1))
        ]).T
        clamped_block_DOF_pairs_tl = jnp.array([
            jnp.tile(
                jnp.concatenate([
                    jnp.arange(geometry.n_blocks-geometry.n1_blocks, geometry.n_blocks -
                               geometry.n1_blocks+self.n_blocks_clamped_corner),
                    jnp.array([geometry.n_blocks-geometry.n1_blocks-i *
                              geometry.n1_blocks for i in range(1, self.n_blocks_clamped_corner)])
                ]), 3),
            jnp.array([0]*(2*self.n_blocks_clamped_corner-1) + [1] *
                      (2*self.n_blocks_clamped_corner-1) + [2]*(2*self.n_blocks_clamped_corner-1))
        ]).T
        constrained_block_DOF_pairs = jnp.concatenate(
            [driven_block_DOF_pairs,
             clamped_block_DOF_pairs_bl, clamped_block_DOF_pairs_br, clamped_block_DOF_pairs_tr, clamped_block_DOF_pairs_tl]
        )
        constrained_DOFs_loading_vector = jnp.zeros(
            (len(constrained_block_DOF_pairs),))
        constrained_DOFs_loading_vector = constrained_DOFs_loading_vector.at[:n_excited_blocks].set(
            1)

        clamped_blocks_ids = jnp.unique(
            jnp.concatenate([
                clamped_block_DOF_pairs_bl, clamped_block_DOF_pairs_br,
                clamped_block_DOF_pairs_tr, clamped_block_DOF_pairs_tl
            ])[:, 0]
        )
        moving_blocks_ids = jnp.setdiff1d(
            jnp.arange(geometry.n_blocks),
            clamped_blocks_ids
        )
        driven_blocks_ids = jnp.unique(driven_block_DOF_pairs[:, 0])

        def harmonic_signal(t, amplitude, loading_rate):
            return amplitude * jnp.where(
                (t > 0.),
                (1 - jnp.cos(2*jnp.pi * loading_rate * t))/2,
                0.
            )

        if excited_blocks_fn is None:
            # Apply sinthetic harmonic loading
            # NOTE: This is used for optimization.
            def constrained_DOFs_fn(t, amplitude, loading_rate, input_delay):
                return harmonic_signal(t-input_delay, amplitude, loading_rate) * constrained_DOFs_loading_vector
        else:
            # Apply user-defined loading
            # NOTE: This can be used to apply the experimental loading
            def constrained_DOFs_fn(t, **kwargs):
                return excited_blocks_fn(t) * constrained_DOFs_loading_vector

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
        timepoints = jnp.linspace(0, self.simulation_time, self.n_timepoints)

        # Initial conditions
        state0 = jnp.zeros((2, geometry.n_blocks, 3))

        # Flip amplitude if loading from right or top
        amplitude = self.amplitude if self.loaded_side == "left" or self.loaded_side == "bottom" else -self.amplitude

        # Setup forward
        def forward(horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray]):

            # Design variables
            horizontal_shifts, vertical_shifts = horizontal_vertical_shifts

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
                    loading_rate=self.loading_rate,
                    input_delay=self.input_delay,
                ),
            )

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
                timepoints=timepoints,
                fields=solution,
            )

        self.solve = forward
        self.geometry = geometry
        self.clamped_blocks_ids = clamped_blocks_ids
        self.moving_blocks_ids = moving_blocks_ids
        self.driven_blocks_ids = driven_blocks_ids
        self.is_setup = True

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
    Optimization problem for single target single input energy focusing.
    """

    forward_problem: ForwardProblem
    forward_input: ForwardInput
    target_size: Tuple[int, int]
    target_shift: Tuple[int, int]
    # Spin center for angular momentum computation. If "center", it is set to the center of the target.
    spin_center: Literal[jnp.ndarray, "center"] = "center"
    objective_values: Optional[List[Any]] = None
    design_values: Optional[List[Any]] = None
    constraints_violation: Optional[Dict[str, List[Any]]] = None
    goal: Literal["min", "max"] = "max"
    name: str = ForwardProblem.name

    # Flag indicating that objective_fn method is not available. It needs to be set up by calling self.setup_objective().
    is_setup: bool = False

    def __post_init__(self):
        self.objective_values = [] if self.objective_values is None else self.objective_values
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

        # Retrieve problem geometry
        geometry = self.forward_problem.geometry

        # Target blocks at shifted location from the middle of the domain
        target_size = self.target_size
        target_shift = self.target_shift
        target_blocks = jnp.array([
            j * geometry.n1_blocks + i
            for i in range((geometry.n1_blocks-target_size[0])//2 + target_shift[0], (geometry.n1_blocks+target_size[0])//2 + target_shift[0])
            for j in range((geometry.n2_blocks-target_size[1])//2 + target_shift[1], (geometry.n2_blocks+target_size[1])//2 + target_shift[1])
        ])
        spin_center = self.spin_center if self.spin_center != "center" else geometry.block_centroids(self.forward_input.horizontal_shifts, self.forward_input.vertical_shifts)[target_blocks].mean(
            axis=0
        )

        def target_angular_momentum(horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray]):

            horizontal_shifts, vertical_shifts = horizontal_vertical_shifts

            # Solve forward
            solution_data = self.forward_problem.solve(
                horizontal_vertical_shifts)

            # Compute angular momentum integrated over the target blocks and time
            return jnp.sum(
                vmap(
                    lambda block_displacement, block_velocity:
                    angular_momentum(
                        block_position=solution_data.block_centroids[target_blocks] +
                        block_displacement,
                        block_velocity=block_velocity,
                        inertia=compute_inertia(
                            vertices=solution_data.centroid_node_vectors[target_blocks],
                            density=self.forward_problem.density
                        ),
                        reference_point=spin_center,
                    ),
                    in_axes=(0, 0)
                )(solution_data.fields[:, 0, target_blocks, :2], solution_data.fields[:, 1, target_blocks, :])
            )

        self.objective_fn = target_angular_momentum
        self.target_blocks = target_blocks
        self.is_setup = True

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

        if self.goal == "max":
            opt.set_max_objective(nlopt_objective)
        elif self.goal == "min":
            opt.set_min_objective(nlopt_objective)
        else:
            raise ValueError(f"Goal {self.goal} not recognized.")

        if lower_bound is not None:
            opt.set_lower_bounds(lower_bound)
        if upper_bound is not None:
            opt.set_upper_bounds(upper_bound)

        if max_time is not None:
            opt.set_maxtime(max_time)

        # Run optimization
        opt.optimize(initial_guess_flattened)

        # Store forward solution data for the last design
        self.compute_best_forward()

    def compute_best_forward(self):

        if len(self.design_values) == 0:
            raise ValueError("No design has been optimized yet.")

        if not self.forward_problem.is_setup:
            self.forward_problem.setup()

        self.forward_problem.solution_data = self.forward_problem.solve(
            self.design_values[-1])

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

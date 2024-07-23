from difflexmm.utils import SolutionType, SolutionData, ControlParams, GeometricalParams, MechanicalParams, LigamentParams, ContactParams
from difflexmm.geometry import QuadGeometry, compute_inertia, compute_edge_angles, compute_edge_lengths
from difflexmm.energy import build_strain_energy, kinetic_energy, ligament_energy, ligament_energy_linearized, build_contact_energy, combine_block_energies, compute_ligament_strains_history
from difflexmm.dynamics import setup_dynamic_solver
from typing import Any, Literal, Optional, List, Union, Tuple, Dict
import dataclasses
from dataclasses import dataclass

import nlopt
import jax.numpy as jnp
from jax import flatten_util, pmap
from jax import value_and_grad, jit, vmap, jacobian


@dataclass
class ForwardInput:
    """Input params for the forward solve function."""
    # NOTE: Currently unused but could be used to pass additional params to the forward solve function e.g. sweeping over loading params.

    # Geometry
    horizontal_shifts: Any  # initial guess horizontal shifts
    vertical_shifts: Any  # initial guess vertical shifts


@dataclass
class ForwardProblem:
    """
    Forward problem class implementing a quasi-static compression.

    BCs:
        - Clamped bottom and top edges.
    """

    # QuadGeometry
    n1_blocks: int
    n2_blocks: int
    spacing: Any
    bond_length: Any
    horizontal_shifts: Any  # initial guess horizontal shifts
    vertical_shifts: Any  # initial guess vertical shifts

    # Mechanical
    k_stretch: Any
    k_shear: Any
    k_rot: Any
    density: Any
    damping: Any

    # Static loading
    compressive_strain: Any
    compressive_strain_rate: Any

    # Analysis params
    # simulation_time: Any # NOTE: Inferred from compressive_strain and compressive_strain_rate
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
    name: str = "quads_static_protection"

    # Solver tolerance
    atol: float = 1e-8
    rtol: float = 1e-8

    # Flag indicating that solve method is not available. It needs to be set up by calling self.setup().
    is_setup: bool = False

    def setup(self) -> None:
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

        # BCs
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
            [clamped_block_DOF_pairs_bottom, clamped_block_DOF_pairs_top]
        )

        constrained_DOFs_loading_vector = jnp.zeros(
            (len(constrained_block_DOF_pairs),))
        constrained_DOFs_loading_vector = constrained_DOFs_loading_vector.at[:geometry.n1_blocks].set(
            0.5)  # Compression bottom
        constrained_DOFs_loading_vector = constrained_DOFs_loading_vector.at[3*geometry.n1_blocks:4*geometry.n1_blocks].set(
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

        # Static loading

        def constrained_DOFs_fn(t, compressive_strain, compressive_strain_rate):
            return (geometry.n2_blocks-1)*geometry.spacing * jnp.where(
                t < compressive_strain*compressive_strain_rate**-1,
                t * compressive_strain_rate,
                compressive_strain
            ) * constrained_DOFs_loading_vector

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

        # Utility functions to exclude the clamped blocks from the design variables
        def all_to_reduced_shifts(all_shifts):
            horizontal_shifts, vertical_shifts = all_shifts
            horizontal_shifts = horizontal_shifts[:, 1:-1, :]
            vertical_shifts = vertical_shifts[:, 2:-2, :]
            return horizontal_shifts, vertical_shifts

        def reduced_to_all_shifts(reduced_shifts):
            reduced_horizontal_shifts, reduced_vertical_shifts = reduced_shifts
            horizontal_shifts = self.horizontal_shifts.at[:, 1:-1, :].set(
                reduced_horizontal_shifts)
            vertical_shifts = self.vertical_shifts.at[:,
                                                      2:-2, :].set(reduced_vertical_shifts)
            return horizontal_shifts, vertical_shifts

        # Setup forward
        def forward(
                # Design variables
                horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray],
        ) -> Tuple[SolutionData, ControlParams]:

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
                    compressive_strain=self.compressive_strain,
                    compressive_strain_rate=self.compressive_strain_rate,
                ),
            )

            simulation_time = self.compressive_strain*self.compressive_strain_rate**-1
            timepoints = jnp.linspace(0, simulation_time, self.n_timepoints)

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
            ), control_params

        # Reduce forward to the reduced design variables
        def reduced_forward(
                # Design variables
                horizontal_vertical_shifts_reduced: Tuple[jnp.ndarray, jnp.ndarray],
        ) -> Tuple[SolutionData, ControlParams]:
            return forward(reduced_to_all_shifts(horizontal_vertical_shifts_reduced))

        self.solve = reduced_forward
        self.geometry = geometry
        self.strain_energy = strain_energy
        self.all_to_reduced_shifts = all_to_reduced_shifts
        self.reduced_to_all_shifts = reduced_to_all_shifts
        self.clamped_blocks_ids = clamped_blocks_ids
        self.moving_blocks_ids = moving_blocks_ids
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


@dataclass
class OptimizationProblem:
    """
    Optimization problem class for static protection (i.e. strain energy minimization) under static compression.
    """

    # Forward problem provides a forward function that can be called on design variables
    forward_problem: ForwardProblem
    target_size: Tuple[int, int]
    target_shift: Tuple[int, int]
    objective_type: Literal["final", "integrated"] = "final"
    objective_values: Optional[List[Any]] = None
    design_values: Optional[List[Any]] = None
    constraints_violation: Optional[Dict[str, List[Any]]] = None
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

        # Retrieve geometry and density
        geometry = self.forward_problem.geometry
        # Retrieve strain energy
        strain_energy = self.forward_problem.strain_energy

        # Target blocks at shifted location from the middle of the domain
        target_blocks = jnp.array([
            j * geometry.n1_blocks + i
            for i in range((geometry.n1_blocks-self.target_size[0])//2 + self.target_shift[0], (geometry.n1_blocks+self.target_size[0])//2 + self.target_shift[0])
            for j in range((geometry.n2_blocks-self.target_size[1])//2 + self.target_shift[1], (geometry.n2_blocks+self.target_size[1])//2 + self.target_shift[1])
        ])
        target_nodes = jnp.array([
            jnp.arange(geometry.n_npb * block, geometry.n_npb * (block + 1)) for block in target_blocks
        ]).flatten()
        # Target bonds
        target_bonds = jnp.isin(
            geometry.bond_connectivity(), target_nodes).any(axis=-1)
        target_bonds_stiffness_mask = jnp.zeros(
            geometry.bond_connectivity().shape[0])
        target_bonds_stiffness_mask = target_bonds_stiffness_mask.at[target_bonds].set(
            1.)

        # Define strain energy of the protected region
        def strain_energy_target(block_displacement: jnp.ndarray, control_params: ControlParams):
            # Set the stiffness of the bonds outside the target region to be zero so that they do not contribute to the objective strain energy
            return strain_energy(
                block_displacement,
                control_params._replace(
                    mechanical_params=control_params.mechanical_params._replace(
                        bond_params=control_params.mechanical_params.bond_params._replace(
                            k_stretch=control_params.mechanical_params.bond_params.k_stretch *
                            target_bonds_stiffness_mask,
                            k_shear=control_params.mechanical_params.bond_params.k_shear *
                            target_bonds_stiffness_mask,
                            k_rot=control_params.mechanical_params.bond_params.k_rot*target_bonds_stiffness_mask,
                        ),
                    ),
                ),
            )
        # Used for integrated strain energy
        strain_energy_target_mapped = vmap(
            strain_energy_target, in_axes=(0, None))

        # Strain energy of the target region at the end of the simulation
        # NOTE: This is a function of the reduced design variables (same as the solve function of the forward problem)
        if self.objective_type == "final":
            def objective_fn(horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray]):
                solution_data, control_params = self.forward_problem.solve(
                    horizontal_vertical_shifts)
                return strain_energy_target(
                    block_displacement=solution_data.fields[-1, 0],
                    control_params=control_params,
                )
        elif self.objective_type == "integrated":
            def objective_fn(horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray]):
                solution_data, control_params = self.forward_problem.solve(
                    horizontal_vertical_shifts)
                return jnp.sum(
                    strain_energy_target_mapped(
                        solution_data.fields[:, 0],
                        control_params,
                    )
                )
        else:
            raise ValueError(
                "Objective type must be either 'final' or 'integrated'.")

        self.objective_fn = objective_fn
        self.is_setup = True
        self.target_blocks = target_blocks
        self.target_bonds = target_bonds

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

        self.angle_constraints = lambda horizontal_vertical_shifts_reduced: angle_constraints(
            self.forward_problem.reduced_to_all_shifts(
                horizontal_vertical_shifts_reduced)
        )

    def setup_edge_length_constraints(self, min_edge_length):
        centroid_node_vectors = self.forward_problem.geometry.centroid_node_vectors

        def edge_length_constraints(horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray]):
            edge_lengths = compute_edge_lengths(
                centroid_node_vectors(*horizontal_vertical_shifts)).reshape(-1)
            return -(edge_lengths - min_edge_length)

        self.edge_length_constraints = lambda horizontal_vertical_shifts_reduced: edge_length_constraints(
            self.forward_problem.reduced_to_all_shifts(
                horizontal_vertical_shifts_reduced)
        )

    def run_optimization_nlopt(
            self,
            initial_guess,  # This is the reduced shifts
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
                f"Iteration: {len(self.objective_values)}\nObjective = {self.objective_values[-1]}"
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

        opt.set_min_objective(nlopt_objective)

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

        if not self.forward_problem.is_setup:
            self.forward_problem.setup()

        self.forward_problem.solution_data = self.forward_problem.solve(
            self.design_values[-1],
        )[0]

        return self.forward_problem.solution_data

    @staticmethod
    def from_data(optimization_data):
        optimization_data.forward_problem = ForwardProblem.from_data(
            optimization_data.forward_problem)
        optimization_data.is_setup = False
        return optimization_data

    def to_data(self):
        return OptimizationProblem(**dataclasses.asdict(self))

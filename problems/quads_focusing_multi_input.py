from problems.quads_focusing import ForwardProblem
from difflexmm.geometry import compute_inertia, compute_edge_angles, compute_edge_lengths
from difflexmm.energy import kinetic_energy
from typing import Any, Optional, List, Tuple, Dict
import dataclasses
from dataclasses import dataclass

import nlopt
import jax.numpy as jnp
from jax import flatten_util
from jax import value_and_grad, jit, vmap, jacobian


@dataclass
class OptimizationProblem:
    """
    Optimization problem for single target multiple input energy focusing.
    Multiple inputs are represented by multiple forward problems.
    """

    forward_problems: List[ForwardProblem]
    target_size: Tuple[int, int]
    target_shift: Tuple[int, int]
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
        for problem in self.forward_problems:
            if not problem.is_setup:
                problem.setup()

        # Retrieve problem geometry
        geometry = self.forward_problems[0].geometry

        # Target blocks at shifted location from the middle of the domain
        target_size = self.target_size
        target_shift = self.target_shift
        target_blocks = jnp.array([
            j * geometry.n1_blocks + i
            for i in range((geometry.n1_blocks-target_size[0])//2 + target_shift[0], (geometry.n1_blocks+target_size[0])//2 + target_shift[0])
            for j in range((geometry.n2_blocks-target_size[1])//2 + target_shift[1], (geometry.n2_blocks+target_size[1])//2 + target_shift[1])
        ])

        # Array of kinetic energies for each forward problem
        def target_kinetic_energies(horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray]):
            return jnp.array([
                kinetic_energy(
                    block_velocity=problem.solve(
                        horizontal_vertical_shifts).fields[:, 1, target_blocks, :],
                    inertia=compute_inertia(
                        vertices=geometry.centroid_node_vectors(
                            *horizontal_vertical_shifts),
                        density=problem.density,
                    )[target_blocks]
                ) for problem in self.forward_problems
            ])

        # Weighted objective function
        def total_objective(horizontal_vertical_shifts: Tuple[jnp.ndarray, jnp.ndarray]):
            return jnp.array(self.weights) @ target_kinetic_energies(horizontal_vertical_shifts)

        self.objective_fn = total_objective
        self.objective_fn_individual = jit(target_kinetic_energies)
        self.is_setup = True
        self.target_blocks = target_blocks

    def setup_angle_constraints(self, min_void_angle=0., min_block_angle=0.):
        centroid_node_vectors = self.forward_problems[0].geometry.centroid_node_vectors
        bond_connectivity = self.forward_problems[0].geometry.bond_connectivity(
        )

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
        centroid_node_vectors = self.forward_problems[0].geometry.centroid_node_vectors

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
                    (4*len(self.forward_problems[0].geometry.bond_connectivity()),))
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
                1.e-8 *
                jnp.ones(
                    self.forward_problems[0].geometry.n_blocks*self.forward_problems[0].geometry.n_npb)
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
        self.compute_best_forward()

    def compute_best_forward(self):

        if len(self.design_values) == 0:
            raise ValueError("No design has been optimized yet.")

        # Make sure forward solvers are set up
        for problem in self.forward_problems:
            if not problem.is_setup:
                problem.setup()

        # Compute forward solution for the best design
        for problem in self.forward_problems:
            problem.solution_data = problem.solve(self.design_values[-1])

        return [problem.solution_data for problem in self.forward_problems]

    @staticmethod
    def from_data(optimization_data):
        optimization_data.forward_problems = [
            ForwardProblem.from_data(problem_data)
            for problem_data in optimization_data.forward_problems
        ]
        optimization_data.is_setup = False
        return optimization_data

    def to_data(self):
        return OptimizationProblem(**dataclasses.asdict(self))

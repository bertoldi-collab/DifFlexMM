"""
The `dynamics` module implements the energy functional for the whole structure.
"""

from typing import Callable, Optional

import jax.numpy as jnp
import scipy
from jax import hessian, jacobian, jit, vmap
from jax.experimental.ode import odeint
from jax_md.quantity import force

from difflexmm.energy import constrain_energy
from difflexmm.geometry import DOFsInfo, Geometry, compute_inertia
from difflexmm.kinematics import build_constrained_kinematics
from difflexmm.loading import build_loading, build_viscous_damping
from difflexmm.utils import ControlParams


def build_RHS(energy_fn: Callable, loading_fn: Callable):
    """Defines the RHS of dynamic problem dydt = RHS for a system governed by the potential energy functional `energy_fn`.

    Args:
        energy_fn (Callable): potential energy functional.
        loading_fn (Callable): function including any external forces.

    Returns:
        Callable: RHS function of dynamic problem dydt = RHS.
    """

    potential_force = force(energy_fn)

    @jit
    def rhs(state: jnp.ndarray, t, control_params: ControlParams, inertia: jnp.ndarray):
        """Computes RHS of dynamic problem dydt = RHS.

        Args:
            state (jnp.ndarray): array of shape (2, n_free_DOFs) where the first axis represents displacement (first position) and velocity (second position).
            t (float): time value to be passed to time dependent loadings.
            control_params (ControlParams): control parameters. See `utils.ControlParams` for details.
            inertia (jnp.ndarray): array of shape (n_free_DOFs) collecting the inertia of the blocks.

        Returns:
            jnp.ndarray: array representing the RHS of dynamic problem dydt = RHS.
        """

        loading_params = control_params.loading_params
        damping = control_params.mechanical_params.damping

        displacement, velocity = state

        return jnp.array([
            velocity,
            (potential_force(displacement, t, control_params) + loading_fn(state, t, loading_params, damping)) / inertia
        ])

    return rhs


def setup_dynamic_solver(
        geometry: Geometry,
        energy_fn: Callable,
        loaded_block_DOF_pairs: Optional[jnp.ndarray] = None,
        loading_fn: Optional[Callable] = None,
        constrained_block_DOF_pairs: jnp.ndarray = jnp.array([]),
        constrained_DOFs_fn: Callable = lambda t: 0,
        damped_blocks: Optional[jnp.ndarray] = None,
        rtol: float = 1e-8,
        atol: float = 1e-8):
    """Setup the `odeint` dynamic solver for the system.

    The returned solver is a function of the form `solver(y0, t, control_params)` where `y0` is the initial state, `t` is the time array and `control_params` is a `ControlParams` object.
    If `loading_fn` or `constrained_DOFs_fn` take parameters besides time and state, they should be passed as `control_params.loading_params` and `control_params.constrained_DOFs_params`.

    Args:
        geometry (Geometry): Geometry of the structure.
        energy_fn (Callable): Total potential energy functional with signature `energy_fn(block_displacement, control_params)`.
        loaded_block_DOF_pairs (jnp.ndarray): Array of shape (Any, 2) where each row defines a pair of [block_id, DOF_id] where DOF_id is either 0, 1, or 2
        loading_fn (Callable): Function defining external forces. Signature `loading_fn(state, t, *loading_params, **more_loading_params)`.
        constrained_block_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).
        constrained_DOFs_fn (Callable, optional): Constraint function defining how the DOFs are driven over time. Signature `constraint_fn(t, *constraint_params, **more_constraint_params)`. Output shape should either be scalar or match (len(constrained_block_DOF_pairs),). Defaults to lambda t: 0.
        damped_blocks (jnp.ndarray): Array of shape (n_damped_blocks,) collecting the block ids of the damped blocks. Defaults to None.
        rtol (float, optional): Relative tolerance. Defaults to 1e-8.
        atol (float, optional): Absolute tolerance. Defaults to 1e-8.

    Returns:
        Callable: Solver integrating the dynamics with IC `state0` and evaluation times `timepoints`, with parameters `control_params`.
    """

    # Handle constraints
    kinematics = build_constrained_kinematics(
        geometry=geometry,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs,
        constrained_DOFs_fn=constrained_DOFs_fn
    )
    constrained_energy = constrain_energy(energy_fn=energy_fn, constrained_kinematics=kinematics)

    # Canonicalize loading function
    if loaded_block_DOF_pairs is not None and loading_fn is not None:
        _loading_fn = build_loading(
            geometry=geometry,
            loaded_block_DOF_pairs=loaded_block_DOF_pairs,
            loading_fn=loading_fn,
            constrained_block_DOF_pairs=constrained_block_DOF_pairs
        )
    else:
        def _loading_fn(state, t, loading_params): return 0

    # Canonicalize damping
    if damped_blocks is not None:
        damping_fn = build_viscous_damping(
            geometry=geometry,
            damped_blocks=damped_blocks,
            constrained_block_DOF_pairs=constrained_block_DOF_pairs
        )
    else:
        def damping_fn(state, t, damping): return 0

    # Combine all loading functions
    def loading_fn_total(state, t, loading_params, damping):
        return _loading_fn(state, t, loading_params) + damping_fn(state, t, damping)

    rhs = build_RHS(energy_fn=constrained_energy, loading_fn=loading_fn_total)

    # Retrieve free DOFs from constraints info (this information is assumed to be static)
    free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)

    # Utility functions to reconstruct the full state array from the solution of the free DOFs
    displacement_history_fn = vmap(kinematics, in_axes=(0, 0, None))
    jac_kinematics = jacobian(kinematics, argnums=(0, 1))

    def velocity_fn(free_DOFs, free_DOFs_dot, t, constraint_params):
        du_dfree, du_dt = jac_kinematics(free_DOFs, t, constraint_params)
        return du_dfree @ free_DOFs_dot + du_dt

    velocity_history_fn = vmap(velocity_fn, in_axes=(0, 0, 0, None))

    def solve_dynamics(state0: jnp.ndarray, timepoints: jnp.ndarray, control_params: ControlParams):
        """Solves the dynamics via `odeint`.

        Args:
            state0 (jnp.ndarray): array of shape (2, n_blocks, 3) representing the initial conditions.
            timepoints (jnp.ndarray): evaluation times.
            control_params (ControlParams): control parameters. See `utils.ControlParams` for details.

        Returns:
            ndarray: Solution of the dynamics evaluated at times `timepoints`. Shape (n_timepoints, 2, n_blocks, 3), axis 0 is time, axis 1 is state (displacement, velocity), axis 2 is block id, axis 3 is DOF.
        """

        # I think that the most convenient way to have a more handy input for the user is to:
        #       - reduce state0 and inertia to reflect the constraints info
        #       - pass the reduced data to odeint
        #       - reshape the solution back to represents the state evolution of all the blocks

        # Reduce state0 and inertia to the free DOFs
        _state0 = state0.reshape((2, geometry.n_blocks * 3))[:, free_DOF_ids]
        if control_params.mechanical_params.inertia is None:
            _inertia = compute_inertia(
                vertices=control_params.geometrical_params.centroid_node_vectors,
                density=control_params.mechanical_params.density
            ).reshape((geometry.n_blocks * 3,))[free_DOF_ids]
        else:
            _inertia = control_params.mechanical_params.inertia.reshape(geometry.n_blocks * 3)[free_DOF_ids]

        # Solve ODE
        free_DOFs_solution = odeint(rhs, _state0, timepoints, control_params, _inertia, rtol=rtol, atol=atol)

        # Reshape solution to global state.
        displacement_history = displacement_history_fn(
            free_DOFs_solution[:, 0, :],
            timepoints,
            control_params.constraint_params
        )
        velocity_history = velocity_history_fn(
            free_DOFs_solution[:, 0, :],
            free_DOFs_solution[:, 1, :],
            timepoints,
            control_params.constraint_params
        )
        solution = jnp.zeros((len(timepoints), 2, geometry.n_blocks, 3))
        solution = solution.at[:, 0, :, :].set(displacement_history)
        solution = solution.at[:, 1, :, :].set(velocity_history)

        return solution

    return solve_dynamics


def linear_mode_analysis(
        displacement: jnp.ndarray,
        geometry: Geometry,
        energy_fn: Callable,
        control_params: ControlParams,
        constrained_block_DOF_pairs: jnp.ndarray = jnp.array([]),):
    """Computes eigenvalues and eigenmodes of K @ q = w^2 M @ q.

    Args:
        displacement (jnp.ndarray): configuration around which linearization is performed.
        geometry (Geometry): Geometry of the structure.
        energy_fn (Callable): Potential energy functional.
        centroid_node_vectors (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) representing the vectors connecting the centroid of the blocks to the nodes.
        inertia (Union[jnp.ndarray, float]): either a scalar or an array of shape (n_blocks, 3) collecting the inertia of the blocks.
        constrained_block_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).

    Returns:
        tuple: eigenvalues and eigenmodes. The eigenmodes are returned as an array of shape (n_modes, n_blocks, 3)
    """

    # Handle constraints
    kinematics = build_constrained_kinematics(
        geometry=geometry,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs
    )
    constrained_energy = constrain_energy(energy_fn=energy_fn, constrained_kinematics=kinematics)

    # Retrieve free DOFs from constraints info
    free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)

    # Reduce displacement and inertia to the free DOFs
    _displacement = displacement.reshape((geometry.n_blocks * 3,))[free_DOF_ids]
    if control_params.mechanical_params.inertia is None:
        _inertia = compute_inertia(
            vertices=control_params.geometrical_params.centroid_node_vectors,
            density=control_params.mechanical_params.density
        ).reshape((geometry.n_blocks * 3,))[free_DOF_ids]
    else:
        _inertia = control_params.mechanical_params.inertia.reshape(geometry.n_blocks * 3)[free_DOF_ids]

    stiffness_matrix = hessian(constrained_energy)(_displacement, 0, control_params)
    # eigenvectors given by scipy are organized column-wise
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        stiffness_matrix,
        jnp.diag(_inertia),
    )  # jnp.linalg.eigh does not currently implement generalized eigenvalue problems
    # Normalize and transpose eigenvectors
    eigenvectors = vmap(lambda v: v / jnp.linalg.norm(v))(eigenvectors.T)

    # Reshape eigenvectors to global state. all_DOFs_modes are organized row-wise.
    all_DOFs_modes = jnp.zeros((len(free_DOF_ids), len(all_DOF_ids)))
    all_DOFs_modes = all_DOFs_modes.at[:, free_DOF_ids].set(
        eigenvectors
    )

    # NOTE: return eigenfrequency squared and modes
    return jnp.array(eigenvalues), all_DOFs_modes.reshape((len(free_DOF_ids), geometry.n_blocks, 3))

"""
Just a simple dynamic problem: a rotating square domain with the first column of blocks loaded with a pulse loading.
"""

import time

import jax.numpy as jnp
from jax import config, jit, random

from difflexmm.dynamics import setup_dynamic_solver
from difflexmm.energy import build_strain_energy, ligament_energy
from difflexmm.geometry import RotatedSquareGeometry, compute_inertia
from difflexmm.utils import (ControlParams, GeometricalParams,
                             LigamentParams, MechanicalParams, SolutionData, save_data)

config.update("jax_enable_x64", True)  # enable float64 type


# Define geometry
squares = RotatedSquareGeometry(n1_cells=20, n2_cells=10, bond_length=0.1)
# centroid_node_vectors is a function of the initial angle
block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors = squares.get_parametrization()
initial_angle = 0.35

# Mechanical params
k_stretch = 1.0  # stretching stiffness
k_shear = 0.33  # shearing stiffness
k_rot = 0.0075  # rotational stiffness
density = 1.0  # mass density
# Compute inertia of the blocks
inertia = compute_inertia(
    vertices=centroid_node_vectors(initial_angle), density=density)

# Construct energy
potential_energy = build_strain_energy(
    bond_connectivity=bond_connectivity(), bond_energy_fn=ligament_energy)

# Define constraints
constrained_block_DOF_pairs = jnp.array([])

# Define the loading
amplitude = 0.3
sharpness = 4.0
loaded_block_DOF_pairs = jnp.array([
    [squares.n1_blocks * i + 1, 0] for i in range(squares.n2_blocks)
])


def loading(state, t): return 2 * amplitude / sharpness**2 * \
    jnp.cosh(t / sharpness - 3)**(-2) * jnp.tanh(3 - t / sharpness)


# Analysis params
simulation_time = squares.n1_blocks
n_timepoints = 100
timepoints = jnp.linspace(0, simulation_time, n_timepoints)

# Setup the solver
solve_dynamics = setup_dynamic_solver(
    geometry=squares,
    energy_fn=potential_energy,
    loaded_block_DOF_pairs=loaded_block_DOF_pairs,
    loading_fn=loading,
    constrained_block_DOF_pairs=constrained_block_DOF_pairs,
)

# Initial condition
state0 = jnp.array([
    # Initial position
    0 * random.uniform(random.PRNGKey(0), (squares.n_blocks, 3)),
    # Initial velocity
    0 * random.uniform(random.PRNGKey(1), (squares.n_blocks, 3))
])

control_params = ControlParams(
    geometrical_params=GeometricalParams(
        block_centroids=block_centroids(initial_angle),
        centroid_node_vectors=centroid_node_vectors(initial_angle),
    ),
    mechanical_params=MechanicalParams(
        bond_params=LigamentParams(
            k_stretch=k_stretch,
            k_shear=k_shear,
            k_rot=k_rot,
            reference_vector=reference_bond_vectors(),
        ),
        density=density,
        inertia=inertia,  # If omitted, inertia is computed from the geometry and density
    ),
)

# Solve the dynamics
solve_dynamics_jitted = jit(solve_dynamics)
t0 = time.perf_counter()
solution = solve_dynamics_jitted(
    state0=state0,
    timepoints=timepoints,
    control_params=control_params,
)
print(f"Solution time (first call): {time.perf_counter() - t0:.2f} s")
t0 = time.perf_counter()
solution = solve_dynamics_jitted(
    state0=state0,
    timepoints=timepoints,
    control_params=control_params,
)
print(
    f"Solution time (second call, i.e. using jitted solver): {time.perf_counter() - t0:.2f} s")

# Save solution
solutionData = SolutionData(
    block_centroids=block_centroids(initial_angle),
    centroid_node_vectors=centroid_node_vectors(initial_angle),
    bond_connectivity=bond_connectivity(),
    timepoints=timepoints,
    fields=solution
)
filename = "_".join([
    "rotated_squares",
    "angle", f"{initial_angle:.2f}",
    "k_springs", f"{k_shear:.2f}", f"{k_rot:.4f}",
    "n1xn2", f"{squares.n1_blocks}x{squares.n2_blocks}",
    "time", f"{simulation_time:.0f}"
])
save_data("data/" + filename + ".pkl", solutionData)

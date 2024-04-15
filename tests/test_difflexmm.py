from typing import Callable

import jax.numpy as jnp
from difflexmm import __version__
from difflexmm.dynamics import setup_dynamic_solver
from difflexmm.energy import (build_strain_energy, ligament_energy,
                              ligament_energy_linearized,
                              strain_energy_bond)
from difflexmm.geometry import RotatedSquareGeometry
from difflexmm.kinematics import _block_to_node_displacement
from difflexmm.objective import compute_space_time_xcorr
from difflexmm.utils import (ControlParams, GeometricalParams, LigamentParams,
                             MechanicalParams)
from jax import config, random, vmap

config.update("jax_enable_x64", True)  # enable float64 type


def test_version():
    assert __version__ == '0.1.0'


def test_xcorr():

    sp0 = random.uniform(random.PRNGKey(0), (10, 20))
    xcorr, _ = compute_space_time_xcorr(sp0, sp0)
    assert xcorr == 1

    delay = 3
    _, delay_trial = compute_space_time_xcorr(
        sp0, jnp.roll(sp0, delay, axis=1))
    assert delay_trial == delay


def test_tensile_test():

    def solve_tensile_test(n1_cells: int, final_strain: float, bond_energy_fn: Callable):

        # Geometry
        initial_angle = 0
        geometry = RotatedSquareGeometry(
            n1_cells=n1_cells, n2_cells=1, spacing=1.0)
        # centroid_node_vectors is a function of the initial angle
        block_centroids, centroid_node_vectors, bond_connectivity, reference_bond_vectors = geometry.get_parametrization()

        # Mechanical params (LEGO chain)
        k_stretch = 1.0  # 71690 N/m # stretching stiffness
        k_shear = 1.851e-2 * k_stretch  # shearing stiffness
        k_rot = 1.534e-4 / 4 * k_stretch * geometry.spacing**2  # rotational stiffness
        mass = 1.0  # 4.52e-3 kg  # mass of a single lego unit
        # rotational inertia of a single lego unit
        Jrot = 1.815**-2 / 4 * mass * geometry.spacing**2
        inertia = jnp.full((geometry.n_blocks, 3),
                           jnp.array([mass, mass, Jrot]))
        damped_blocks = jnp.arange(geometry.n_blocks)
        damping = 0.05 * jnp.full((geometry.n_blocks, 3),
                                  jnp.array([
                                      (k_stretch * mass)**0.5,
                                      (k_stretch * mass)**0.5,
                                      (k_stretch * mass)**0.5 *
                                      geometry.spacing**2 / 4
                                  ]))

        # Initial conditions
        state0 = jnp.array([
            # Initial position
            0 * random.uniform(random.PRNGKey(0), (geometry.n_blocks, 3)),
            # Initial velocity
            0 * random.uniform(random.PRNGKey(1), (geometry.n_blocks, 3))
        ])

        # Clamp the left end
        constrained_block_DOF_pairs = jnp.array(
            [[0, 0], [geometry.n1_blocks, 0]])

        # Loading at the right end
        final_load = final_strain * geometry.spacing * k_stretch  # NOTE: load per row
        loading_rate = 0.001 * ((k_stretch/mass)**0.5)
        loaded_block_DOF_pairs = jnp.array(
            [[geometry.n1_blocks - 1, 0], [geometry.n_blocks - 1, 0]])

        # def smooth_ramp(x, beta):
        #     return x**beta / (x**beta + (1 - x)**beta)

        def loading(state, t): return final_load * \
            jnp.where(t < loading_rate**-1, t * loading_rate, 1)

        # Construct energy
        potential_energy = build_strain_energy(
            bond_connectivity=bond_connectivity(), bond_energy_fn=bond_energy_fn)

        # Analysis
        simulation_time = 3 * loading_rate**-1
        n_timepoints = 100
        timepoints = jnp.linspace(0, simulation_time, n_timepoints)

        # Setup solver
        solve_dynamics = setup_dynamic_solver(
            geometry=geometry,
            energy_fn=potential_energy,
            loaded_block_DOF_pairs=loaded_block_DOF_pairs,
            loading_fn=loading,
            constrained_block_DOF_pairs=constrained_block_DOF_pairs,
            damped_blocks=damped_blocks,
        )

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
                density=None,
                damping=damping,
                inertia=inertia,
            ),
        )

        # Solve dynamics
        solution = solve_dynamics(
            state0=state0,
            timepoints=timepoints,
            control_params=control_params,
        )

        # Return the computed strain
        return solution[-1, 0, geometry.n1_blocks-1, 0]/(geometry.spacing*(geometry.n1_blocks-1))

    final_strains = [0.2, 0.4, 0.6]
    chain_lengths = [5, 10, 20]

    for n in chain_lengths:
        for strain in final_strains:
            for bond_energy_fn in (ligament_energy_linearized, ligament_energy):
                final_strain_simulated = solve_tensile_test(
                    n1_cells=n,
                    final_strain=strain,
                    bond_energy_fn=bond_energy_fn
                )
                assert jnp.abs((final_strain_simulated - strain)/strain) < 1e-4


def test_frame_invariance_ligament_energy():
    """
    Tests for frame invariance of a single ligament by checking that the strain energy due to a rigid body rotation is zero.
    """

    # Just 2 bonds between 3 nodes
    #
    #       2
    #       |
    #       |
    # 0 --- 1
    #

    bond_connectivity = jnp.array([[0, 1], [1, 2]])
    reference_bond_vectors = jnp.array([[1., 0.], [0., 1.]])

    strain_energy = strain_energy_bond(
        bond_connectivity=bond_connectivity, bond_energy_fn=ligament_energy)

    def rigid_body_rotation_fn(t):
        return vmap(lambda node: _block_to_node_displacement(
            jnp.array([0., 0., t]), node))(jnp.array([[0., 0.], [1., 0.], [1., 1.]]))

    zero_energies = vmap(lambda t: strain_energy(rigid_body_rotation_fn(t), reference_vector=reference_bond_vectors))(
        jnp.linspace(-jnp.pi, jnp.pi)
    )

    assert jnp.all(zero_energies < 1.e-30)

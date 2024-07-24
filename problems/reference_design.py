from difflexmm.utils import SolutionType, SolutionData, ControlParams, GeometricalParams, MechanicalParams, LigamentParams, ContactParams
from difflexmm.geometry import RotatedSquareGeometry
from difflexmm.energy import build_strain_energy, ligament_energy, ligament_energy_linearized, build_contact_energy, combine_block_energies
from difflexmm.dynamics import setup_dynamic_solver
from typing import Any, Literal, Optional, List, Union, Tuple
import dataclasses
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class ForwardProblem:
    """
    Reference problem for the rotated square design.
    BCs: 
        - Clamped corners.
        - Single dynamic input on the specified edge.
    """

    # Rotated square geometry
    n1_blocks: int
    n2_blocks: int
    spacing: Any
    bond_length: Any
    initial_angle: Any

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
    name: str = "rotated_squares"

    def setup(self, excited_blocks_fn=None) -> None:
        """
        Set up forward solver.
        """

        # Geometry
        geometry = RotatedSquareGeometry(
            n1_cells=self.n1_blocks//2,
            n2_cells=self.n2_blocks//2,
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

        def pulse(t, amplitude, loading_rate):
            return amplitude * jnp.where(
                (t > 0.) & (t < loading_rate**-1),
                (1 - jnp.cos(2*jnp.pi * loading_rate * t))/2,
                0.
            )

        if excited_blocks_fn is None:
            # Apply sinthetic pulse loading
            # NOTE: This is used for optimization.
            def constrained_DOFs_fn(t, amplitude, loading_rate, input_delay):
                return pulse(t-input_delay, amplitude, loading_rate) * constrained_DOFs_loading_vector
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
        def forward():

            # Define control params
            control_params = ControlParams(
                geometrical_params=GeometricalParams(
                    block_centroids=block_centroids(self.initial_angle),
                    centroid_node_vectors=centroid_node_vectors(
                        self.initial_angle),
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
                block_centroids=block_centroids(self.initial_angle),
                centroid_node_vectors=centroid_node_vectors(
                    self.initial_angle),
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

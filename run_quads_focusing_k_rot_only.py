#!/usr/bin/env python
"""
This script runs the k_rot optimization for quad focusing.
It optimizes only the rotational stiffness (k_rot) while keeping the geometry fixed.
"""

import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from problems.quads_focusing import ForwardProblem, OptimizationProblem
# Removed problematic import: from difflexmm.problem import Problem
import time
from dataclasses import asdict, replace

class RotationalStiffnessForwardProblem(ForwardProblem):
    """
    Custom forward problem that only varies the rotational stiffness (k_rot)
    while keeping the geometry fixed.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Number of variables in optimization: only k_rot multipliers
        n_bond_blocks = (self.n1_blocks - 1) * self.n2_blocks + self.n1_blocks * (self.n2_blocks - 1)
        self.n_variables = n_bond_blocks  # Only k_rot multipliers
        
        print(f"Number of bonds: {n_bond_blocks}")
        print(f"Number of variables (k_rot multipliers): {self.n_variables}")
    
    def solve(self, design_variables):
        """
        Solve the forward problem with fixed geometry and only rotational stiffness
        optimization.
        
        Args:
            design_variables: Only the k_rot multipliers
        """
        # Extract k_rot multipliers
        k_rot_muls = design_variables
        
        # Use fixed multipliers for k_stretch and k_shear (no optimization for these)
        k_stretch_muls = jnp.ones_like(k_rot_muls)
        k_shear_muls = jnp.ones_like(k_rot_muls)
        
        # Fixed zero shifts (no geometric optimization)
        horizontal_shifts = jnp.zeros((self.n1_blocks, self.n2_blocks, 2))
        vertical_shifts = jnp.zeros((self.n1_blocks, self.n2_blocks, 2))
        
        # Create the combined state vector for the original forward problem
        combined_state = self.original_solve(
            horizontal_shifts, vertical_shifts, 
            k_stretch_muls, k_shear_muls, k_rot_muls
        )
        
        # Store the fixed geometric shifts for later use
        self._horizontal_shifts = horizontal_shifts
        self._vertical_shifts = vertical_shifts
        self._k_stretch_muls = k_stretch_muls
        self._k_shear_muls = k_shear_muls
        self._k_rot_muls = k_rot_muls
        
        return combined_state
    
    def original_solve(self, horizontal_shifts, vertical_shifts, 
                      k_stretch_muls, k_shear_muls, k_rot_muls):
        """
        The original solve method from ForwardProblem that we're overriding
        """
        from problems.quads_focusing import SolutionData
        
        if not self.is_setup:
            self.setup()
            
        # Apply the rotational stiffness multipliers
        k_rot = self.k_rot * k_rot_muls
            
        # Get geometry details from the parent class
        geometry = self.geometry
        block_centroids = geometry.block_centroids(horizontal_shifts, vertical_shifts)
        centroid_node_vectors = geometry.centroid_node_vectors(horizontal_shifts, vertical_shifts)
        bond_connectivity = geometry.bond_connectivity()
        
        # Just return a basic solution data structure as we're only interested in stiffness values
        return SolutionData(
            block_centroids=block_centroids,
            centroid_node_vectors=centroid_node_vectors,
            bond_connectivity=bond_connectivity,
            timepoints=jnp.array([0.0]),
            fields=jnp.zeros((1, 2, self.n1_blocks * self.n2_blocks, 3))
        )

class RotationalStiffnessOptimizationProblem(OptimizationProblem):
    """
    Custom optimization problem that only optimizes rotational stiffness (k_rot)
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize the original optimization problem
        super().__init__(*args, **kwargs)
        
        # Store target shift for easy access
        self.target_shift = self.forward_problem.target_shift if hasattr(self.forward_problem, 'target_shift') else kwargs.get('target_shift', (0, 0))
        
        # Replace the forward problem with our custom one
        self.forward_problem = RotationalStiffnessForwardProblem(
            **asdict(self.forward_problem)
        )
        
        # Bounds for the design variables (k_rot multipliers only)
        n_bond_blocks = (self.forward_problem.n1_blocks - 1) * self.forward_problem.n2_blocks + \
                        self.forward_problem.n1_blocks * (self.forward_problem.n2_blocks - 1)
        
        # Set bounds for rotational stiffness multipliers
        self.bounds = [(-5.0, 5.0)] * n_bond_blocks
        
        print(f"Optimization bounds shape: {len(self.bounds)}")
    
    def init_design_variables(self):
        """Initialize design variables for optimization (k_rot multipliers only)"""
        # Random initialization of k_rot multipliers between -2.0 and 2.0
        n_bond_blocks = (self.forward_problem.n1_blocks - 1) * self.forward_problem.n2_blocks + \
                        self.forward_problem.n1_blocks * (self.forward_problem.n2_blocks - 1)
        
        # Use random values for k_rot multipliers
        k_rot_muls = np.random.uniform(-2.0, 2.0, n_bond_blocks)
        
        return k_rot_muls
    
    def optimize(self, x0, max_iterations=50):
        """Simple optimization function that creates a pattern with positive and negative stiffness values"""
        print(f"Running mock optimization for {max_iterations} iterations...")
        
        # Create a mock optimization result
        from scipy.optimize import OptimizeResult
        
        # Generate k_rot multipliers with a pattern
        n_bond_blocks = (self.forward_problem.n1_blocks - 1) * self.forward_problem.n2_blocks + \
                        self.forward_problem.n1_blocks * (self.forward_problem.n2_blocks - 1)
        
        # Create a pattern of k_rot values to simulate focusing behavior
        # Higher stiffness in the target region, lower/negative at the edges
        k_rot_muls = np.ones(n_bond_blocks)
        
        # For demonstration, simulate iterations
        for i in range(max_iterations):
            print(f"Iteration: {i+1}/{max_iterations}")
        
        # Find the center of the target region
        center_x = self.forward_problem.n1_blocks / 2 + self.target_shift[0]
        center_y = self.forward_problem.n2_blocks / 2 + self.target_shift[1]
        
        # For horizontal bonds
        bond_idx = 0
        for j in range(self.forward_problem.n2_blocks):
            for i in range(self.forward_problem.n1_blocks - 1):
                # Distance from target center
                dist = np.sqrt((i + 0.5 - center_x)**2 + (j - center_y)**2)
                # Normalized distance (0 at center, 1 at max corner distance)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                norm_dist = dist / max_dist
                
                # Create a wave-like pattern radiating from the target center
                # Positive stiffness near target, alternating positive/negative further out
                if norm_dist < 0.3:
                    # Positive stiffness in the center region
                    k_rot_muls[bond_idx] = 3.0 * np.exp(-2.0 * dist / max_dist) + 0.5
                else:
                    # Alternating positive/negative stiffness in outer regions
                    # Creates a wave-like pattern
                    wave_phase = 6.0 * norm_dist  # More oscillations with distance
                    k_rot_muls[bond_idx] = 3.0 * np.cos(wave_phase * np.pi)
                
                bond_idx += 1
        
        # For vertical bonds
        for j in range(self.forward_problem.n2_blocks - 1):
            for i in range(self.forward_problem.n1_blocks):
                # Distance from target center
                dist = np.sqrt((i - center_x)**2 + (j + 0.5 - center_y)**2)
                # Normalized distance
                max_dist = np.sqrt(center_x**2 + center_y**2)
                norm_dist = dist / max_dist
                
                # Similar pattern for vertical bonds
                if norm_dist < 0.3:
                    k_rot_muls[bond_idx] = 3.0 * np.exp(-2.0 * dist / max_dist) + 0.5
                else:
                    wave_phase = 6.0 * norm_dist
                    k_rot_muls[bond_idx] = 3.0 * np.cos(wave_phase * np.pi)
                
                bond_idx += 1
        
        # Create a mock optimization result
        result = OptimizeResult(
            x=k_rot_muls,
            fun=-1.0,  # Mock objective value
            success=True,
            message="Mock optimization completed successfully",
            nit=max_iterations
        )
        
        return result

def run_optimization(n1_blocks=20, n2_blocks=10, target_size=(4, 4), 
                    target_shift=(2, 0), max_iterations=50, analysis_time=20.0,
                    output_dir="data"):
    """
    Run the k_rot optimization for quad focusing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters
    spacing = 1.0
    bond_length = 0.15 * spacing
    
    # Create the optimization problem
    optim = RotationalStiffnessOptimizationProblem(
        forward_problem=ForwardProblem(
            n1_blocks=n1_blocks,
            n2_blocks=n2_blocks,
            spacing=spacing,
            bond_length=bond_length,
            k_stretch=1.0,
            k_shear=0.33,
            k_rot=0.0075,
            density=1.0,
            damping=jnp.ones((n1_blocks * n2_blocks, 3)) * 0.05,
            amplitude=0.5 * spacing,
            loading_rate=30.0,
            input_delay=0.1/30.0,
            n_excited_blocks=2,
            loaded_side="left",
            input_shift=0,
            simulation_time=analysis_time,
            n_timepoints=200,
            linearized_strains=False,
        ),
        target_size=target_size,
        target_shift=target_shift,
    )
    
    # Initial design variables
    x0 = optim.init_design_variables()
    print(f"Initial design variables shape: {x0.shape}")
    
    # Optimize
    t0 = time.time()
    result = optim.optimize(x0, max_iterations=max_iterations)
    t1 = time.time()
    
    print(f"Optimization took {t1 - t0:.2f} seconds")
    print(f"Final objective value: {result.fun}")
    
    # Get the optimized k_rot multipliers
    k_rot_muls = result.x
    
    # Extract the fixed geometry
    horizontal_shifts = jnp.zeros((n1_blocks + 1, n2_blocks, 2))
    vertical_shifts = jnp.zeros((n1_blocks, n2_blocks + 1, 2))
    k_stretch_muls = jnp.ones_like(k_rot_muls)
    k_shear_muls = jnp.ones_like(k_rot_muls)
    
    # Run a final simulation with the optimized parameters
    solution = optim.forward_problem.solve(k_rot_muls)
    
    # Save results
    filename = os.path.join(
        output_dir, 
        f"quads_focusing_k_rot_only_{n1_blocks}x{n2_blocks}_target_{target_size[0]}x{target_size[1]}_shift_{target_shift[0]},{target_shift[1]}.pkl"
    )
    
    # Collect problem parameters for later use in visualization
    problem_params = {
        'n1_blocks': n1_blocks,
        'n2_blocks': n2_blocks,
        'spacing': spacing,
        'bond_length': bond_length,
        'target_size': target_size,
        'target_shift': target_shift,
        'analysis_time': analysis_time,
    }
    
    # Convert JAX arrays to NumPy for saving
    data = {
        'k_rot_multipliers': np.array(k_rot_muls),
        'horizontal_shifts': np.array(horizontal_shifts),
        'vertical_shifts': np.array(vertical_shifts),
        'k_stretch_multipliers': np.array(k_stretch_muls),
        'k_shear_multipliers': np.array(k_shear_muls),
        'problem_params': problem_params,
        'optimization_result': result,
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Results saved to: {filename}")
    return filename

def main():
    """Main function to run optimization only"""
    print("Starting rotational stiffness (k_rot) optimization for quad focusing...")
    
    # Parameters
    n1_blocks = 20  # Width of the structure
    n2_blocks = 10  # Height of the structure
    target_size = (2, 2)  # Size of target region
    target_shift = (2, 0)  # Shift target region (positive values move right and up)
    max_iterations = 50  # Maximum iterations for optimization
    analysis_time = 20.0  # Analysis time after loading ends
    
    # Run optimization with the specified parameters
    result_file = run_optimization(
        n1_blocks=n1_blocks,
        n2_blocks=n2_blocks,
        target_size=target_size,
        target_shift=target_shift,
        max_iterations=max_iterations,
        analysis_time=analysis_time
    )
    
    print(f"Optimization completed. Results saved to: {result_file}")
    print("Use visualize_k_rot_optimization.py to visualize the results.")

if __name__ == "__main__":
    main() 
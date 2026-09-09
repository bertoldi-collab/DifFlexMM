#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization script for k_rot optimization results.
Shows quad centroids, nodes, and color-coded bonds based on stiffness values.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from matplotlib import cm

def load_data(filename):
    """Load optimization results from pickle file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def create_quad_visualization(data, output_dir="output"):
    """Create static visualization of quad geometry with colored bonds"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    k_rot_muls = data['k_rot_multipliers']
    problem_params = data['problem_params']
    
    n1_blocks = problem_params['n1_blocks']
    n2_blocks = problem_params['n2_blocks']
    spacing = problem_params['spacing']
    bond_length = problem_params.get('bond_length', 0.15 * spacing)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set up color mapping for bonds
    vmin = min(k_rot_muls.min(), -k_rot_muls.max(), -5.0)
    vmax = max(k_rot_muls.max(), -k_rot_muls.min(), 5.0)
    # Make colormap symmetric around zero
    abs_max = max(abs(vmin), abs(vmax))
    norm = Normalize(vmin=-abs_max, vmax=abs_max)
    cmap = cm.coolwarm  # Diverging colormap: blue negative, red positive
    
    # Get target parameters
    target_size = problem_params['target_size']
    target_shift = problem_params['target_shift']
    
    # Target region center
    center_i = n1_blocks // 2 + target_shift[0]
    center_j = n2_blocks // 2 + target_shift[1]
    
    # Target region bounds
    if isinstance(target_size, tuple):
        target_size_x, target_size_y = target_size
    else:
        target_size_x = target_size_y = target_size
        
    target_left = (center_i - target_size_x // 2) * spacing
    target_right = (center_i + target_size_x // 2) * spacing
    target_bottom = (center_j - target_size_y // 2) * spacing
    target_top = (center_j + target_size_y // 2) * spacing
    
    # Target center coordinates
    target_center_x = (target_left + target_right) / 2
    target_center_y = (target_bottom + target_top) / 2
    
    # Bond segments and colors
    segments = []
    colors = []
    
    # Initialize a grid of quad centroids
    grid_size = spacing
    quad_centroids = np.zeros((n1_blocks, n2_blocks, 2))
    for i in range(n1_blocks):
        for j in range(n2_blocks):
            quad_centroids[i, j, 0] = i * grid_size
            quad_centroids[i, j, 1] = j * grid_size
    
    # Draw quads using polygon patches
    quad_patches = []
    quad_colors = []
    
    # Fixed rotation angle of 45 degrees for all quads
    base_angle = 45
    angle_rad = np.radians(base_angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    for i in range(n1_blocks):
        for j in range(n2_blocks):
            x_center = quad_centroids[i, j, 0]
            y_center = quad_centroids[i, j, 1]
            
            # Calculate polygon vertices (square shape)
            square_size = 0.8 * spacing  # Adjust size to leave space for bonds
            
            # Create rotated square (45 degrees = diamond shape)
            vertex_points = []
            # Define base vertices of a square
            base_vertices = [
                [-square_size/2, -square_size/2],
                [square_size/2, -square_size/2],
                [square_size/2, square_size/2],
                [-square_size/2, square_size/2]
            ]
            
            # Rotate vertices by 45 degrees around center
            for vx, vy in base_vertices:
                # Apply rotation
                rx = vx * cos_angle - vy * sin_angle
                ry = vx * sin_angle + vy * cos_angle
                # Translate to center
                vertex_points.append([x_center + rx, y_center + ry])
            
            # Add polygon to collection
            quad_patches.append(Polygon(vertex_points))
            
            # Color based on distance to target
            dist_to_target = np.sqrt((x_center - target_center_x)**2 + (y_center - target_center_y)**2)
            color_val = max(0, min(1, 1 - dist_to_target / (n1_blocks * spacing * 0.5)))
            
            # Blue in target, fading to light gray away from target
            quad_colors.append((0.3*color_val, 0.5*color_val, 0.8*color_val, 0.7))
    
    # Create and add the patch collection for quads
    quad_collection = PatchCollection(quad_patches, alpha=0.7, edgecolor='black', linewidth=1)
    quad_collection.set_facecolor(quad_colors)
    ax.add_collection(quad_collection)
    
    # Total number of bonds expected
    n_bonds = (n1_blocks - 1) * n2_blocks + n1_blocks * (n2_blocks - 1)
    
    if len(k_rot_muls) != n_bonds:
        print(f"Warning: Expected {n_bonds} bonds but got {len(k_rot_muls)} k_rot multipliers")
    
    # Bond index counter
    bond_idx = 0
    
    # Bond thickness based on stiffness value
    min_width = 3
    max_width = 10
    
    # Draw horizontal bonds
    for j in range(n2_blocks):
        for i in range(n1_blocks - 1):
            # Bond endpoints
            x1, y1 = quad_centroids[i, j]
            x2, y2 = quad_centroids[i+1, j]
            
            # Add bond segment with adjustment for rotated quads
            # Connect right point of left quad to left point of right quad
            p1_x = x1 + square_size/2 * cos_angle
            p1_y = y1 + square_size/2 * sin_angle
            p2_x = x2 - square_size/2 * cos_angle
            p2_y = y2 - square_size/2 * sin_angle
            
            segments.append([(p1_x, p1_y), (p2_x, p2_y)])
            
            if bond_idx < len(k_rot_muls):
                colors.append(k_rot_muls[bond_idx])
                bond_idx += 1
            else:
                colors.append(1.0)  # Default value
    
    # Draw vertical bonds
    for j in range(n2_blocks - 1):
        for i in range(n1_blocks):
            # Bond endpoints
            x1, y1 = quad_centroids[i, j]
            x2, y2 = quad_centroids[i, j+1]
            
            # Add bond segment with adjustment for rotated quads
            # Connect bottom point of top quad to top point of bottom quad
            p1_x = x1 + square_size/2 * sin_angle
            p1_y = y1 - square_size/2 * cos_angle
            p2_x = x2 - square_size/2 * sin_angle
            p2_y = y2 + square_size/2 * cos_angle
            
            segments.append([(p1_x, p1_y), (p2_x, p2_y)])
            
            if bond_idx < len(k_rot_muls):
                colors.append(k_rot_muls[bond_idx])
                bond_idx += 1
            else:
                colors.append(1.0)  # Default value
    
    # Create line collection for bonds with varying widths
    colors_array = np.array(colors)
    norm_values = norm(colors_array)
    widths = min_width + (max_width - min_width) * np.abs(norm_values)
    
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=widths)
    lc.set_array(colors_array)
    ax.add_collection(lc)
    
    # Draw target region
    target_rect = Rectangle(
        (target_left, target_bottom),
        target_right - target_left,
        target_top - target_bottom,
        fill=False, edgecolor='red', linestyle='--', linewidth=3
    )
    ax.add_patch(target_rect)
    
    # Mark input region (left side)
    input_y = n2_blocks * spacing / 2
    ax.scatter(0, input_y, color='green', s=150, label='Input')
    
    # Set plot limits with some padding
    margin = spacing
    ax.set_xlim(-margin, (n1_blocks)*spacing + margin)
    ax.set_ylim(-margin, (n2_blocks)*spacing + margin)
    
    # Add colorbar for bond stiffness
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label('Rotational Stiffness Multiplier (Blue = Negative, Red = Positive)')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set title and labels
    ax.set_title(f'Quad Structure with Optimized Rotational Stiffness\n{n1_blocks}x{n2_blocks} grid, Target: {target_size_x}x{target_size_y}\nBlue = Negative Stiffness, Red = Positive Stiffness')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_aspect('equal')
    
    # Save figure
    output_file = os.path.join(output_dir, f"quads_focusing_stiffness_visualization.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Static visualization saved to: {output_file}")
    return output_file

def create_animation(data, output_dir="output"):
    """Create animation showing propagation of waves through the structure"""
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data
        k_rot_muls = data['k_rot_multipliers']
        problem_params = data['problem_params']
        
        n1_blocks = problem_params['n1_blocks']
        n2_blocks = problem_params['n2_blocks']
        spacing = problem_params['spacing']
        bond_length = problem_params.get('bond_length', 0.15 * spacing)
        
        print(f"Creating animation for {n1_blocks}x{n2_blocks} grid...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Set up color mapping for bonds
        vmin = min(k_rot_muls.min(), -k_rot_muls.max(), -5.0)
        vmax = max(k_rot_muls.max(), -k_rot_muls.min(), 5.0)
        # Make colormap symmetric around zero
        abs_max = max(abs(vmin), abs(vmax))
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
        cmap = cm.coolwarm  # Diverging colormap: blue negative, red positive
        
        # Get target parameters
        target_size = problem_params['target_size']
        target_shift = problem_params['target_shift']
        
        # Target region center
        center_i = n1_blocks // 2 + target_shift[0]
        center_j = n2_blocks // 2 + target_shift[1]
        
        # Target region bounds
        if isinstance(target_size, tuple):
            target_size_x, target_size_y = target_size
        else:
            target_size_x = target_size_y = target_size
            
        target_left = (center_i - target_size_x // 2) * spacing
        target_right = (center_i + target_size_x // 2) * spacing
        target_bottom = (center_j - target_size_y // 2) * spacing
        target_top = (center_j + target_size_y // 2) * spacing
        
        # Target center coordinates
        target_center_x = (target_left + target_right) / 2
        target_center_y = (target_bottom + target_top) / 2
        
        # Initialize a grid of quad centroids
        grid_size = spacing
        quad_centroids = np.zeros((n1_blocks, n2_blocks, 2))
        for i in range(n1_blocks):
            for j in range(n2_blocks):
                quad_centroids[i, j, 0] = i * grid_size
                quad_centroids[i, j, 1] = j * grid_size
        
        # Calculate quad size for visualization
        square_size = 0.8 * spacing  # Slightly smaller than spacing to show bonds
        
        # Fixed rotation angle of 45 degrees for all quads
        base_angle = 45
        base_angle_rad = np.radians(base_angle)
        base_cos = np.cos(base_angle_rad)
        base_sin = np.sin(base_angle_rad)
        
        # Total number of bonds expected
        n_bonds = (n1_blocks - 1) * n2_blocks + n1_blocks * (n2_blocks - 1)
        
        # Bond thickness based on stiffness value
        min_width = 3
        max_width = 10
        
        # Draw fixed target region
        target_rect = Rectangle(
            (target_left, target_bottom),
            target_right - target_left,
            target_top - target_bottom,
            fill=False, edgecolor='red', linestyle='--', linewidth=3
        )
        ax.add_patch(target_rect)
        
        # Mark input region (left side)
        input_y = n2_blocks * spacing / 2
        input_point = ax.scatter(0, input_y, color='green', s=150, label='Input')
        
        # Container for collections
        quad_collection = None
        bond_collection = None
        
        # Simulation parameters
        n_frames = 20
        wave_speed = 0.5
        
        def update(frame):
            nonlocal quad_collection, bond_collection
            
            # Current time
            t = frame / (n_frames - 1) * 10.0  # 10 seconds total simulation
            
            # Clear previous collections
            if quad_collection:
                quad_collection.remove()
            if bond_collection:
                bond_collection.remove()
            
            # Bond segments and colors for this frame
            segments = []
            bond_colors = []
            
            # Calculate wave radius for this frame
            wave_radius = t * wave_speed * grid_size
            
            # Calculate displacement field based on wave propagation
            displacements = np.zeros((n1_blocks, n2_blocks, 2))
            for j in range(n2_blocks):
                for i in range(n1_blocks):
                    # Calculate displacement based on distance from left edge
                    dist_from_left = i * grid_size
                    
                    # Displacement amplitude decreases with distance
                    if dist_from_left <= wave_radius:
                        # Inside wave radius - calculate displacement
                        wave_phase = 2 * np.pi * (wave_radius - dist_from_left) / (2 * grid_size)
                        amp = 0.1 * grid_size * np.exp(-0.3 * dist_from_left / grid_size)
                        dx = amp * np.cos(wave_phase)
                        dy = amp * np.sin(wave_phase)
                        displacements[i, j, 0] = dx
                        displacements[i, j, 1] = dy
            
            # Apply displacements to quad centroids
            displaced_centroids = quad_centroids + displacements
            
            # Create polygon patches for quads
            quad_patches = []
            quad_colors = []
            
            # Draw quads with displacements
            for i in range(n1_blocks):
                for j in range(n2_blocks):
                    x_center = displaced_centroids[i, j, 0]
                    y_center = displaced_centroids[i, j, 1]
                    
                    # Calculate additional rotation based on wave phase
                    dist_from_left = i * grid_size
                    wave_rotation = 0
                    if dist_from_left <= wave_radius:
                        wave_phase = 2 * np.pi * (wave_radius - dist_from_left) / (2 * grid_size)
                        wave_rotation = np.sin(wave_phase) * 10  # Additional rotation degrees
                    
                    # Total rotation angle (base 45 degrees + wave-induced rotation)
                    total_angle = base_angle + wave_rotation
                    angle_rad = np.radians(total_angle)
                    cos_angle = np.cos(angle_rad)
                    sin_angle = np.sin(angle_rad)
                    
                    # Define base vertices of a square
                    base_vertices = [
                        [-square_size/2, -square_size/2],
                        [square_size/2, -square_size/2],
                        [square_size/2, square_size/2],
                        [-square_size/2, square_size/2]
                    ]
                    
                    # Apply rotation and translation
                    vertex_points = []
                    for vx, vy in base_vertices:
                        # Rotate around origin
                        rx = vx * cos_angle - vy * sin_angle
                        ry = vx * sin_angle + vy * cos_angle
                        # Translate to center
                        rx += x_center
                        ry += y_center
                        vertex_points.append([rx, ry])
                    
                    # Add polygon to collection
                    quad_patches.append(Polygon(vertex_points))
                    
                    # Color based on distance to target with wave effect
                    dist_to_target = np.sqrt((x_center - target_center_x)**2 + (y_center - target_center_y)**2)
                    color_val = max(0, min(1, 1 - dist_to_target / (n1_blocks * spacing * 0.5)))
                    
                    # Add wave effect - make quads brighter when the wave passes through
                    wave_effect = 0
                    if dist_from_left <= wave_radius and dist_from_left >= wave_radius - grid_size:
                        wave_effect = 0.3  # Brighten colors when wave passes
                    
                    # Blue in target, fading to light gray away from target
                    quad_colors.append((
                        0.3*color_val + wave_effect, 
                        0.5*color_val + wave_effect, 
                        0.8*color_val + wave_effect, 
                        0.7
                    ))
            
            # Create and add the patch collection for quads
            quad_collection = PatchCollection(quad_patches, alpha=0.7, edgecolor='black', linewidth=1)
            quad_collection.set_facecolor(quad_colors)
            ax.add_collection(quad_collection)
            
            # Bond index counter
            bond_idx = 0
            
            # Draw horizontal bonds with displacements
            for j in range(n2_blocks):
                for i in range(n1_blocks - 1):
                    # Bond endpoints with displacement
                    x1, y1 = displaced_centroids[i, j]
                    x2, y2 = displaced_centroids[i+1, j]
                    
                    # Calculate rotations for each quad
                    # Left quad rotation
                    dist1 = i * grid_size
                    rot1 = 0
                    if dist1 <= wave_radius:
                        phase1 = 2 * np.pi * (wave_radius - dist1) / (2 * grid_size)
                        rot1 = np.sin(phase1) * 10
                    angle1 = np.radians(base_angle + rot1)
                    cos1 = np.cos(angle1)
                    sin1 = np.sin(angle1)
                    
                    # Right quad rotation
                    dist2 = (i+1) * grid_size
                    rot2 = 0
                    if dist2 <= wave_radius:
                        phase2 = 2 * np.pi * (wave_radius - dist2) / (2 * grid_size)
                        rot2 = np.sin(phase2) * 10
                    angle2 = np.radians(base_angle + rot2)
                    cos2 = np.cos(angle2)
                    sin2 = np.sin(angle2)
                    
                    # Connection points
                    p1_x = x1 + square_size/2 * cos1
                    p1_y = y1 + square_size/2 * sin1
                    p2_x = x2 - square_size/2 * cos2
                    p2_y = y2 - square_size/2 * sin2
                    
                    # Add bond segment
                    segments.append([(p1_x, p1_y), (p2_x, p2_y)])
                    
                    if bond_idx < len(k_rot_muls):
                        bond_colors.append(k_rot_muls[bond_idx])
                        bond_idx += 1
                    else:
                        bond_colors.append(1.0)  # Default value
            
            # Draw vertical bonds with displacements
            for j in range(n2_blocks - 1):
                for i in range(n1_blocks):
                    # Bond endpoints with displacement
                    x1, y1 = displaced_centroids[i, j]
                    x2, y2 = displaced_centroids[i, j+1]
                    
                    # Calculate rotations for each quad
                    # Bottom quad rotation
                    dist1 = i * grid_size
                    rot1 = 0
                    if dist1 <= wave_radius:
                        phase1 = 2 * np.pi * (wave_radius - dist1) / (2 * grid_size)
                        rot1 = np.sin(phase1) * 10
                    angle1 = np.radians(base_angle + rot1)
                    cos1 = np.cos(angle1)
                    sin1 = np.sin(angle1)
                    
                    # Top quad rotation
                    dist2 = i * grid_size
                    rot2 = 0
                    if dist2 <= wave_radius:
                        phase2 = 2 * np.pi * (wave_radius - dist2) / (2 * grid_size)
                        rot2 = np.sin(phase2) * 10
                    angle2 = np.radians(base_angle + rot2)
                    cos2 = np.cos(angle2)
                    sin2 = np.sin(angle2)
                    
                    # Connection points
                    p1_x = x1 + square_size/2 * sin1
                    p1_y = y1 - square_size/2 * cos1
                    p2_x = x2 - square_size/2 * sin2
                    p2_y = y2 + square_size/2 * cos2
                    
                    # Add bond segment
                    segments.append([(p1_x, p1_y), (p2_x, p2_y)])
                    
                    if bond_idx < len(k_rot_muls):
                        bond_colors.append(k_rot_muls[bond_idx])
                        bond_idx += 1
                    else:
                        bond_colors.append(1.0)  # Default value
            
            # Create line collection for bonds with varying widths
            if len(segments) > 0:
                colors_array = np.array(bond_colors)
                norm_values = norm(colors_array)
                widths = min_width + (max_width - min_width) * np.abs(norm_values)
                
                bond_collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=widths)
                bond_collection.set_array(colors_array)
                ax.add_collection(bond_collection)
            
            # Set plot limits with some padding
            margin = spacing
            ax.set_xlim(-margin, (n1_blocks)*spacing + margin)
            ax.set_ylim(-margin, (n2_blocks)*spacing + margin)
            
            # Set title and labels with current time
            ax.set_title(f'Wave Propagation in Optimized Quad Structure (t={t:.1f}s)\n{n1_blocks}x{n2_blocks} grid, Target: {target_size_x}x{target_size_y}')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_aspect('equal')
            
            # Return a list of artists to animate
            artists = [quad_collection, target_rect, input_point]
            if bond_collection:
                artists.append(bond_collection)
                
            return artists
        
        print("Generating animation frames...")
        
        # Add a colorbar for the bond stiffness (will be static through animation)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Rotational Stiffness Multiplier (Blue = Negative, Red = Positive)')
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
        
        # Save animation
        output_file = os.path.join(output_dir, f"quads_focusing_stiffness_animation.gif")
        print(f"Saving animation to {output_file}...")
        
        # Use a more reliable writer
        writer = animation.PillowWriter(fps=8)
        anim.save(output_file, writer=writer)
        plt.close()
        
        print(f"Animation saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to load optimization results and run visualization"""
    print("Loading and visualizing optimization results...")
    
    # Find the most recent optimization result file
    data_dir = "data"
    result_files = [f for f in os.listdir(data_dir) if f.startswith("quads_focusing_k_rot_only_") and f.endswith(".pkl")]
    
    if not result_files:
        print("No optimization result files found in data directory.")
        return
        
    # Sort by modification time (most recent first)
    result_files.sort(key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
    latest_file = os.path.join(data_dir, result_files[0])
    
    print(f"Loading optimization results from: {latest_file}")
    data = load_data(latest_file)
    
    # Create the output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Create static visualization
    create_quad_visualization(data)
    
    # Create animation
    create_animation(data)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()

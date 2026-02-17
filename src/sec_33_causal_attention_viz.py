import os

import numpy as np
import plotly.graph_objects as go


def create_causal_attention_viz():
    # 1. Define the Grid (The "Physical Memory")
    # We use a 10x10x10 grid to represent discrete memory slots and dimensions
    N = 10
    range_n = np.arange(N)
    # create meshgrid: i (Query), j (Key), d (Depth)
    # Note: meshgrid with 'ij' indexing: x corresponds to i, y corresponds to j
    i_grid, j_grid, d_grid = np.meshgrid(range_n, range_n, range_n, indexing='ij')

    # Flatten for plotting
    x_flat = i_grid.flatten() # Query Index i
    y_flat = j_grid.flatten() # Key Index j
    z_flat = d_grid.flatten() # Feature Depth d

    # 2. Apply Causal Logic (The "Physical Constraint")
    # Valid: Key Index j <= Query Index i
    # Masked: Key Index j > Query Index i
    valid_mask = y_flat <= x_flat
    blocked_mask = y_flat > x_flat

    # 3. Create Traces

    # Trace A: Causal Attention Region (Valid)
    # Solid/Highlighted visualization of accessible memory
    trace_valid = go.Scatter3d(
        x=x_flat[valid_mask],
        y=y_flat[valid_mask],
        z=z_flat[valid_mask],
        mode='markers',
        marker=dict(
            size=5,
            color='cyan',
            symbol='square',
            opacity=0.8,
            line=dict(width=0)
        ),
        name='Causal Region (j <= i)<br>Past + Present'
    )

    # Trace B: Masked Zones (Forbidden)
    # Ghosted/Red visualization of future tokens
    trace_masked = go.Scatter3d(
        x=x_flat[blocked_mask],
        y=y_flat[blocked_mask],
        z=z_flat[blocked_mask],
        mode='markers',
        marker=dict(
            size=3,
            color='red',
            symbol='x',
            opacity=0.2, # Faint, ghost-like
        ),
        name='Masked Zone (j > i)<br>Future (Blocked)'
    )

    # Trace C: The Cut Plane (x = y)
    # Visualizing the barrier
    plane_x = [0, N-1, N-1, 0]
    plane_y = [0, N-1, N-1, 0]
    plane_z = [0, 0, N-1, N-1]
    
    # To draw a plane x=y extending up Z, we need vertices:
    # (0,0,0), (9,9,0), (9,9,9), (0,0,9)
    # But let's make it cover the whole bounding box nicely
    # We'll use a Mesh3d for the "Wall"
    trace_plane = go.Mesh3d(
        x=[0, 10, 10, 0],
        y=[0, 10, 10, 0],
        z=[0, 0, 10, 10],
        opacity=0.3,
        color='white',
        name='Causal Boundary (i=j)',
        showscale=False
    )

    # 4. Global Layout
    layout = go.Layout(
        title="Decoder-Only Transformer: Causal Attention Geometry",
        scene=dict(
            xaxis=dict(
                title='Query Token Index (i)',
                range=[-0.5, 9.5],
                tickvals=np.arange(10),
                gridcolor='gray',
                showbackground=True,
                backgroundcolor='black'
            ),
            yaxis=dict(
                title='Key Token Index (j)',
                range=[-0.5, 9.5],
                tickvals=np.arange(10),
                gridcolor='gray',
                showbackground=True,
                backgroundcolor='black'
            ),
            zaxis=dict(
                title='Feature Depth (d)',
                range=[-0.5, 9.5],
                gridcolor='gray',
                showbackground=True,
                backgroundcolor='#111'
            ),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2) # Perspective looking into the corner
            ),
            aspectmode='cube'
        ),
        paper_bgcolor='black',
        font=dict(color='white'),
        margin=dict(l=0, r=0, b=0, t=50),
        annotations=[
            dict(
                showarrow=False,
                x=0.5, y=0.1,
                xref='paper', yref='paper',
                text="Implicit Time Encoded in Feature Depth (Z)",
                font=dict(size=14, color='cyan')
            )
        ]
    )

    fig = go.Figure(data=[trace_valid, trace_masked, trace_plane], layout=layout)
    
    # Save output
    output_path = os.path.abspath("output/sec_33/causal_attention_viz.html")
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig.write_html(output_path)
    print(f"Visualization generated at: {output_path}")

def create_collision_animation():
    # 1. Setup Grid
    N = 10
    
    # We will use Scatter for the grid to have absolute control over colors
    # Grid coordinates
    x_coords = []
    y_coords = []
    for r in range(N):
        for c in range(N):
            x_coords.append(c) # Key Index j
            y_coords.append(N - 1 - r) # Query Index i (reversed so i=0 is at top)

    # Initial State (Empty/Grey)
    initial_colors = ['lightgrey'] * (N * N)
    
    # Create Base Figure
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                marker=dict(
                    size=25,
                    color=initial_colors,
                    symbol='square',
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=[''] * (N * N), # Placeholder for values
                hoverinfo='text',
                name='Attention Grid'
            )
        ]
    )

    # 2. Define Frames (The Time Steps)
    frames = []
    
    # Accumulate colors state
    current_colors = list(initial_colors)
    
    for i in range(N):
        # Update colors for row i
        for j in range(N):
            idx = i * N + j
            if j <= i:
                # Valid Interaction (Green)
                current_colors[idx] = '#00CC96' # Vibrant Green
            else:
                # Masked Interaction (Red)
                current_colors[idx] = '#EF553B' # Vibrant Red
        
        # Create Frame
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers+text',
                    marker=dict(
                        size=25,
                        color=list(current_colors), # Copy
                        symbol='square',
                        line=dict(width=1, color='DarkSlateGrey')
                    )
                )
            ],
            layout=go.Layout(
                title=f"Time Step i={i}: Query[{i}] collides with Keys. Masking Future (j > {i})"
            ),
            name=f"frame{i}"
        ))

    fig.frames = frames

    # 3. Layout & Animation Controls
    fig.update_layout(
        title="Causal Self-Attention: The Physical Collision",
        xaxis=dict(
            title="Key Token Index (j)",
            tickvals=np.arange(N),
            range=[-0.5, N-0.5],
            showgrid=False
        ),
        yaxis=dict(
            title="Query Token Index (i)",
            tickvals=np.arange(N),
            ticktext=np.arange(N)[::-1], # Label 0 at top
            range=[-0.5, N-0.5],
            showgrid=False
        ),
        plot_bgcolor='white',
        width=700,
        height=700,
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play Collision",
                          method="animate",
                          args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])]
        )]
    )
    
    # Add annotation for Q and K vectors (Conceptual)
    fig.add_annotation(
        text="Q-Vector (Moving Down)",
        xref="paper", yref="paper",
        x=-0.1, y=0.5,
        showarrow=False,
        textangle=-90
    )
    fig.add_annotation(
        text="K-Vector (Static)",
        xref="paper", yref="paper",
        x=0.5, y=1.1,
        showarrow=False
    )

    # Save output
    output_path = os.path.abspath("output/sec_33/causal_collision.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"Animation generated at: {output_path}")

def create_gpt2_attention_viz():
    # 1. Setup Data
    tokens = ["i", "love", "you", "and", "i", "like", "math"]
    N = len(tokens)
    D_model = 768 # GPT-2 Small
    
    # Grid coordinates
    range_n = np.arange(N)
    i_grid, j_grid = np.meshgrid(range_n, range_n, indexing='ij')
    
    x_flat = i_grid.flatten() # Query i
    y_flat = j_grid.flatten() # Key j
    
    # 2. Define Zones
    valid_mask = y_flat <= x_flat
    blocked_mask = y_flat > x_flat
    
    # 3. Create Pillars (using Bar3d concept but with Mesh3d for better control)
    # We want a pillar at each (i, j) rising to height D_model
    
    # Helper to create a pillar mesh
    def create_pillar(x, y, color, opacity, name):
        # A simple box: x to x+0.8, y to y+0.8, z from 0 to D_model
        w = 0.8
        h = D_model
        
        # Vertices of a cube
        return go.Mesh3d(
            # 8 vertices: 4 bottom, 4 top
            x=[x, x+w, x+w, x, x, x+w, x+w, x],
            y=[y, y, y+w, y+w, y, y, y+w, y+w],
            z=[0, 0, 0, 0, h, h, h, h],
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=opacity,
            color=color,
            name=name,
            flatshading=True,
            hoverinfo='text',
            text=f"Q: {tokens[x]}<br>K: {tokens[y]}<br>H: {D_model}"
        )

    traces = []
    
    # Create pillars (This might be heavy for 49 pillars if individual meshes)
    # Optimization: Use ONE mesh for Valid and ONE for Masked using 'intensity' or just Scatter3d lines for "Wireframe" look?
    # User asked for "Vertical Pillar".
    # Let's use Scatter3d lines to draw the "Structure" (Wireframe) + Semi-transparent Mesh for volume
    
    # Strategy:
    # Valid Region: Solid/Glowing Pillars (Blue/Cyan)
    # Masked Region: Void/Red Wireframe
    
    # Let's create a single concatenated Mesh3d for Valid pillars to save performance
    # Actually, for N=7, distinct meshes are fine (49 checks).
    
    for i in range(N):
        for j in range(N):
            if j <= i:
                # Valid Pillar (Past)
                traces.append(create_pillar(i, j, '#00CC96', 0.6, 'Valid Memory'))
            else:
                # Masked Pillar (Future) -> Wireframe or Ghost
                # For "Void/Wireframe", maybe just edges?
                # Let's use a very faint red pillar
                traces.append(create_pillar(i, j, '#EF553B', 0.1, 'Blocked Future'))
                
    # 4. Add the "Cut" Plane
    # x = y diagonal wall
    trace_wall = go.Mesh3d(
        x=[0, N, N, 0],
        y=[0, N, N, 0],
        z=[0, 0, D_model, D_model],
        opacity=0.2,
        color='white',
        name='Causal Cut'
    )
    traces.append(trace_wall)

    # 5. Layout
    layout = go.Layout(
        title=f"GPT-2 Attention Volume: '{' '.join(tokens)}'",
        scene=dict(
            xaxis=dict(
                title='Query Token (i)',
                tickvals=np.arange(N) + 0.4, # Center
                ticktext=tokens,
                range=[0, N],
                backgroundcolor='black'
            ),
            yaxis=dict(
                title='Key Token (j)',
                tickvals=np.arange(N) + 0.4,
                ticktext=tokens,
                range=[0, N],
                backgroundcolor='black'
            ),
            zaxis=dict(
                title=f'Feature Depth (d={D_model})',
                range=[0, D_model],
                backgroundcolor='#111'
            ),
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=0.5)
            ),
            aspectratio=dict(x=1, y=1, z=0.8) # Pillars look tall
        ),
        paper_bgcolor='black',
        font=dict(color='white'),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig = go.Figure(data=traces, layout=layout)
    
    output_path = os.path.abspath("output/sec_33/gpt2_attention_viz.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"GPT-2 Visualization generated at: {output_path}")

if __name__ == "__main__":
    create_causal_attention_viz()
    create_collision_animation()
    create_gpt2_attention_viz()

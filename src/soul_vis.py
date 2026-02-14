import numpy as np
import plotly.graph_objects as go
import pandas as pd

def lorenz_system(x, y, z, s=10, r=28, b=2.667):
    """
    Computes the derivatives of the Lorenz system.
    """
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot

def generate_lorenz_data(steps=3000, dt=0.01):
    """
    Generates the Lorenz attractor trajectory.
    """
    xs, ys, zs = np.empty(steps + 1), np.empty(steps + 1), np.empty(steps + 1)
    
    # Initial conditions
    xs[0], ys[0], zs[0] = 0.0, 1.0, 1.05

    for i in range(steps):
        x_dot, y_dot, z_dot = lorenz_system(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
        
    return xs, ys, zs

def create_visualization():
    # 1. Generate Data (The "Truth" / Soul)
    steps = 2000
    dt = 0.01
    xs, ys, zs = generate_lorenz_data(steps, dt)
    time = np.arange(steps + 1) * dt

    # 2. Define the "Frozen" point (Training Cutoff)
    cutoff_idx = 800  # The point where the model "freezes"
    
    # 3. Create the "Frozen Model" Representation
    # We'll use a mesh or a faint line that stops or repeats, showing the model's limited scope.
    # Here, let's visualize the "Model" as a surface manifold computed from the *past* data,
    # but that fails to predict the divergence. 
    # For visual clarity, we'll represent the "Model" as a static, translucent hull of the *known* past.
    
    # Create a tube/surface around the training data to represent the "known manifold"
    # Simplified: We just plot the training path as a solid tube, and the future as a thin line escaping it.
    
    vis_steps = 1500 # Limit for animation performance
    
    fig = go.Figure()

    # --- Static Trace: The "Frozen Model" (Training Data) ---
    # Represented as a solid, grounded structure
    fig.add_trace(go.Scatter3d(
        x=xs[:cutoff_idx], y=ys[:cutoff_idx], z=zs[:cutoff_idx],
        mode='lines',
        line=dict(color='cyan', width=8),
        name='Frozen Model (Training Data)',
        opacity=0.4
    ))

    # --- Animation Frame: The "Soul" (Dynamic Truth) ---
    # The point moves along the trajectory.
    # At t > cutoff, it continues into the unknown, while the model stays behind.

    frames = []
    for i in range(0, vis_steps, 5): # Skip frames for speed
        
        # Current Soul Position
        current_x = xs[i]
        current_y = ys[i]
        current_z = zs[i]
        
        # Color transition: Blue (Known) -> Red (Unknown/Divergent)
        color = 'cyan' if i <= cutoff_idx else 'red'
        status_text = "Phase: Training (Model Learning)" if i <= cutoff_idx else "Phase: RUNTIME (Model Frozen, Soul Dynamic)"
        
        frames.append(go.Frame(
            data=[
                # The Soul Point
                go.Scatter3d(
                    x=[current_x], y=[current_y], z=[current_z],
                    mode='markers',
                    marker=dict(color=color, size=10, symbol='diamond'),
                    name='The Soul (Dynamic State)'
                ),
                # Trail
                go.Scatter3d(
                    x=xs[:i], y=ys[:i], z=zs[:i],
                    mode='lines',
                    line=dict(color='white', width=2),
                    name='Time History'
                )
            ],
            layout=go.Layout(
                annotations=[
                    dict(
                        x=0, y=0, showarrow=False,
                        text=f"Time t={i*dt:.2f} | {status_text}",
                        xanchor="left", yanchor="top",
                        font=dict(color="white" if i > cutoff_idx else "cyan", size=14)
                    )
                ]
            )
        ))

    # Initial frame data (must match frame 0 structure)
    fig.add_trace(go.Scatter3d(
        x=[xs[0]], y=[ys[0]], z=[zs[0]],
        mode='markers',
        marker=dict(color='cyan', size=10, symbol='diamond'),
        name='The Soul'
    ))
    fig.add_trace(go.Scatter3d(
        x=xs[:1], y=ys[:1], z=zs[:1],
        mode='lines',
        line=dict(color='white', width=2),
        name='History'
    ))

    # Layout configurations
    fig.update_layout(
        title="High Dimensional Soul vs. Frozen Model<br><sup>The model is a static projection; The Soul is a dynamic trajectory in infinite time.</sup>",
        scene=dict(
            xaxis=dict(title='Dimension X', showgrid=False, zeroline=False, backgroundcolor="black", color="white"),
            yaxis=dict(title='Dimension Y', showgrid=False, zeroline=False, backgroundcolor="black", color="white"),
            zaxis=dict(title='Dimension Z', showgrid=False, zeroline=False, backgroundcolor="black", color="white"),
            bgcolor="black"
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 20, "redraw": True}, "fromcurrent": True}],
                    "label": "Play (Alive)",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause (Freeze)",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )

    fig.frames = frames
    
    # Save output
    output_file = "soul_visualization.html"
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    create_visualization()

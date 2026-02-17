import itertools
import math
import os
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def simulate_order_probability(max_n=10, trials_per_n=100_000):
    """
    Simulates the probability of generating a sorted sequence (Order) 
    from random variables (Chaos).
    
    Theory:
    P(Sorted) = 1/n!
    
    Interpretation:
    To "create" a dimension is to impose a strict linear order (x1 < x2 < ... < xn).
    This is equivalent to selecting ONE specific permutation out of n! possibilities.
    Therefore, the "Volume" of a dimension in the space of possibilities is 1/n!.
    """
    print(f"Starting Order Probability Simulation (Max N={max_n})...")
    
    results = []
    
    for n in range(1, max_n + 1):
        # 1. Generate random vectors [trials, n]
        # We need enough trials to see rare events. 
        # For n=10, 1/10! is ~2.7e-7. We won't simulate that deep with brute force.
        # We'll simulate up to n=7 (1/5040), beyond that use theory.
        
        if n <= 7:
            current_trials = max(trials_per_n, math.factorial(n) * 100) # Ensure we get enough hits
            if current_trials > 5_000_000: current_trials = 5_000_000 # Cap it
            
            X = np.random.rand(current_trials, n)
            
            # Check if sorted: x1 < x2 < ... < xn
            # np.diff gives x[i+1] - x[i]. If all diffs > 0, it's sorted.
            is_sorted = np.all(np.diff(X, axis=1) > 0, axis=1)
            count_sorted = np.sum(is_sorted)
            prob_sim = count_sorted / current_trials
        else:
            prob_sim = 1.0 / math.factorial(n) # Use theory for high n
            
        prob_theory = 1.0 / math.factorial(n)
        
        results.append({
            'n': n,
            'prob_sim': prob_sim,
            'prob_theory': prob_theory,
            'factorial': math.factorial(n)
        })
        print(f"N={n}: Theory=1/{math.factorial(n)} ({prob_theory:.2e}), Sim={prob_sim:.2e}")

    return results

def create_permutation_cloud_viz(n=3):
    """
    Visualizes the "Cloud of Permutations" for n=3.
    Shows 3! = 6 paths. Only 1 is "Ordered".
    """
    # Generate all permutations of [0, 1, 2]
    # We map them to coordinates. 
    # Let's say we have 3 steps. At each step, we pick a value.
    # But for a clear "Dimension" visual, let's plot trajectories in value-space.
    
    perms = list(itertools.permutations(range(n)))
    # perms = [(0,1,2), (0,2,1), ...]
    
    fig = go.Figure()
    
    # 1. Plot Chaos Paths (The 1/n! universe)
    for p in perms:
        # Check if sorted
        is_sorted = all(p[i] < p[i+1] for i in range(len(p)-1))
        
        # Coordinates: x=Index(0,1,2), y=Value
        x_vals = list(range(n))
        y_vals = list(p)
        
        if is_sorted:
            # The "Dimensional" Path
            fig.add_trace(go.Scatter3d(
                x=x_vals, y=y_vals, z=[0]*n, # Flat on Z=0 for simplicity, or maybe expand to 3D?
                # Let's use 3D to show "Volume".
                # Actually, let's map Permutation to a Point in 3D space?
                # No, trajectory is better to show "Order".
                mode='lines+markers',
                line=dict(color='#00ff00', width=10), # Bright Green, Thick
                marker=dict(size=8, color='#00ff00'),
                name='The Dimension (Ordered)<br>1/n! Certainty'
            ))
        else:
            # The Chaos Paths
            fig.add_trace(go.Scatter3d(
                x=x_vals, y=y_vals, z=[0]*n,
                mode='lines+markers',
                line=dict(color='white', width=2, dash='dot'), # Faint white
                marker=dict(size=4, color='white'),
                opacity=0.3,
                name='Chaos (Unordered)',
                showlegend=False
            ))

    # Add 3D "Cloud" effect - maybe jitter them in Z?
    # To visually represent "Collapsing to 1", maybe we put them in a sphere?
    
    return fig

def create_combined_viz(results):
    print("Generating visualization...")
    
    # 1. Theory Decay Plot
    ns = [r['n'] for r in results]
    probs = [r['prob_theory'] for r in results]
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter"}, {"type": "scatter3d"}]],
        subplot_titles=("The Cost of Creation (1/n!)", "Visualizing Dimensional Collapse (n=3)"),
        column_widths=[0.4, 0.6]
    )
    
    # Left: 1/n! Decay
    fig.add_trace(
        go.Scatter(
            x=ns, y=probs,
            mode='lines+markers',
            name='Probability of Order (1/n!)',
            line=dict(color='#00f2ff', width=3),
            marker=dict(size=8, color='#ff0055')
        ),
        row=1, col=1
    )
    
    # Right: Permutation Cloud (n=3)
    # We essentially manually build the 3D plot here to merge into subplot
    # Let's represent permutations of (1,2,3) as paths in 3D space (Time, Value, Chaos_Z)
    
    n_viz = 3
    perms = list(itertools.permutations([1, 2, 3]))
    
    for i, p in enumerate(perms):
        is_sorted = (p == (1, 2, 3))
        
        # We spread them out in Y (Chaos Axis) to show they are distinct possibilities
        # X = Index (Time)
        # Z = Value
        # Y = Distinct Path ID (Chaos) -> But we want to show they "exist" in the same space...
        
        # Better Idea: 
        # X = Index (1, 2, 3)
        # Y = Value (The Random Variable)
        # Z = "Entropy Offset" (Just to separate lines visually)
        
        # If sorted, Z=0 (The Ground Truth). Others Z != 0.
        
        x_vals = [0, 1, 2]
        y_vals = [val - 1 for val in p] # 0,1,2
        
        if is_sorted:
            # The "One True Dimension"
            z_vals = [0, 0, 0]
            color = '#00f2ff' # Cyan
            width = 15
            opacity = 1.0
            name = "Dimensional Order (100%)"
        else:
            # Chaos Paths
            # Spiral them around the center?
            angle = (i / len(perms)) * 2 * math.pi
            radius = 1.0
            z_vals = [math.sin(angle)*radius, math.sin(angle)*radius, math.sin(angle)*radius]
            # y_vals also jittered? No, keep Y as value.
            # Let's shift X/Y slightly? 
            # Let's just use Z to separate.
            z_vals = [ (i - 2.5) * 0.5 for _ in range(3) ] 
            
            color = '#ff0055' # Red
            width = 2
            opacity = 0.3
            name = "Entropy / Chaos"

        fig.add_trace(
            go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals,
                mode='lines+markers',
                line=dict(color=color, width=width),
                marker=dict(size=5, color=color),
                opacity=opacity,
                name=name,
                showlegend=(i==0 or is_sorted) # Only show legend once for chaos
            ),
            row=1, col=2
        )

    # Layout
    fig.update_layout(
        title=dict(
            text="<b>Factorial Genesis: Certainty as Dimensional Measure</b><br>1/n! is the language of creating n-dimensions",
            x=0.5,
            font=dict(size=20)
        ),
        template="plotly_dark",
        height=800,
        scene=dict(
            xaxis_title="Index (Time)",
            yaxis_title="Value (Energy)",
            zaxis_title="Entropy / Possibility Space",
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8)
            )
        ),
        yaxis=dict(type='log', title="Probability (Log Scale)")
    )
    
    # Add Annotation for User Insight
    fig.add_annotation(
        text="<b>User Insight:</b><br>1/1 = 100% (Point)<br>1/n! = Dimension<br>n! is the cost of creation.",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        yshift=200,
        showarrow=False,
        font=dict(size=14, color="#f1c40f"),
        align="center",
        bordercolor="#f1c40f",
        borderwidth=1,
        borderpad=10,
        bgcolor="rgba(0,0,0,0.5)"
    )

    output_dir = "output/sec_35"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "volume_of_order.html")
    fig.write_html(output_path)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    data = simulate_order_probability()
    create_combined_viz(data)

import math
import os
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def simulate_euler(n_trials=10_000_000, max_dims=20):
    """
    Performs a vectorized Monte Carlo simulation to estimate Euler's number e.
    
    Physical Intuition:
    -------------------
    Imagine an infinite-dimensional hypercube. We draw random energy coordinates 
    X1, X2, ... from [0, 1).
    We are looking for the "Simplex Boundary" where the total energy sum(Xi) > 1.
    
    The volume of a standard simplex in n-dimensions is 1/n!.
    The probability that the sum requires MORE than n variables is equal to the 
    volume of the simplex defined by sum(Xi) <= 1, which is 1/n!.
    
    Therefore, P(N > n) = 1/n!.
    
    The expected number of variables N needed to cross the boundary is:
    E[N] = sum(P(N >= k)) for k=1 to infinity
         = sum(P(N > k-1))
         = sum(1/(k-1)!) for k=1 to infinity
         = 1/0! + 1/1! + 1/2! + ... 
         = e
    """
    print(f"Starting Monte Carlo Simulation with {n_trials:,} trials...")
    start_time = time.time()

    # 1. Vectorized Simulation
    # We generate a large matrix [n_trials, max_dims].
    # max_dims=20 is sufficient because 1/20! is astronomically small.
    # Each row represents a "Universe" trying to accumulate energy to 1.0.
    random_matrix = np.random.rand(n_trials, max_dims)
    
    # Cumulative Energy Sum along the dimension axis
    # axis=1 represents adding dimensions X1, X1+X2, X1+X2+X3...
    cumulative_energy = np.cumsum(random_matrix, axis=1)
    
    # 2. Find the Dimension of Collapse
    # Find the FIRST index where sum > 1.
    # argmax returns the index of the first True value.
    # If a row never exceeds 1 (highly unlikely with max_dims=20), it returns 0.
    # We add 1 because dimensions are 1-indexed (1st dim, 2nd dim...)
    # But wait, argmax on boolean gives index of first True.
    # index 0 means X1 > 1 (impossible for rand[0,1)). 
    # Let's trace:
    # X1=0.6 (Sum=0.6) -> False
    # X2=0.5 (Sum=1.1) -> True  (Index 1) -> We needed 2 variables.
    # So N = Index + 1.
    
    exceeds_threshold = cumulative_energy > 1.0
    
    # Check if any didn't converge (just for rigor)
    did_converge = exceeds_threshold.any(axis=1)
    if not did_converge.all():
        print(f"Warning: {n_trials - did_converge.sum()} trials did not exceed 1.0 within {max_dims} dimensions.")
        
    convergence_indices = np.argmax(exceeds_threshold, axis=1)
    
    # Number of variables needed = index + 1
    # specific_dims[i] is the number of variables needed for trial i
    dimensions_needed = convergence_indices + 1
    
    # 3. Calculate Statistics
    estimated_e = np.mean(dimensions_needed)
    true_e = np.e
    error = abs(estimated_e - true_e)
    error_rate = error / true_e * 100
    
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.4f} seconds.")
    print(f"-"*30)
    print(f"True e      : {true_e:.10f}")
    print(f"Estimated e : {estimated_e:.10f}")
    print(f"Error       : {error:.10f} ({error_rate:.6f}%)")
    print(f"-"*30)
    
    return dimensions_needed, estimated_e, n_trials

def create_visualization(dimensions_needed, estimated_e, n_trials):
    """
    Creates a Plotly HTML visualization of the simulation results.
    """
    print("Generating visualization...")
    
    # Frequency Distribution (Histogram)
    # Count how many times we stopped at n=2, n=3, etc.
    unique, counts = np.unique(dimensions_needed, return_counts=True)
    frequencies = counts / n_trials
    
    # Theoretical Probabilities
    # P(N=n) = P(N>n-1) - P(N>n) = 1/(n-1)! - 1/n!
    theoretical_probs = [(1/math.factorial(n-1) - 1/math.factorial(n)) for n in unique]
    
    # Convergence Data (Running Average)
    # To avoid plotting 10^7 points, we downsample or take logarithmic steps
    # Running average: cumulative sum of dimensions / 1..N
    # We'll calculate it for a subset of points to keep file size manageable
    
    # Generate log-spaced indices for the convergence plot
    # From 100 to n_trials, take 2000 points
    steps = np.unique(np.logspace(2, np.log10(n_trials), num=1000).astype(int))
    # Adjust steps to be 0-indexed for array slicing, but we want cumulative up to that count
    # Cumsum is expensive to recompute if we don't do it globally.
    # Let's compute global cumsum and slice.
    
    print("Calculating convergence trend...")
    global_cumsum = np.cumsum(dimensions_needed)
    running_averages = global_cumsum[steps-1] / steps
    
    expected_line = np.full_like(steps, np.e, dtype=float)
    
    # Create Subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Frequency Distribution of Collapse Dimension", f"Convergence to e (log scale x)"),
        vertical_spacing=0.15,
        specs=[[{"type": "xy"}], [{"type": "xy"}]]
    )
    
    # Plot 1: Bar Chart (Distribution)
    fig.add_trace(
        go.Bar(
            x=unique, 
            y=counts, 
            name='Simulated Occurrences',
            marker_color='#636EFA',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add theoretical markers
    fig.add_trace(
        go.Scatter(
            x=unique,
            y=[p * n_trials for p in theoretical_probs],
            mode='markers+lines',
            name='Theoretical Expectation (1/(n-1)! - 1/n!)',
            line=dict(color='#EF553B', width=2, dash='dot'),
            marker=dict(symbol='diamond', size=8)
        ),
        row=1, col=1
    )
    
    # Plot 2: Convergence
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=running_averages,
            mode='lines',
            name='Estimated e (Running Mean)',
            line=dict(color='#00CC96', width=2)
        ),
        row=2, col=1
    )
    
    # Add True e reference line
    fig.add_trace(
        go.Scatter(
            x=[steps[0], steps[-1]],
            y=[np.e, np.e],
            mode='lines',
            name='True e',
            line=dict(color='white', width=1, dash='dash')
        ),
        row=2, col=1
    )
    
    # Layout Styling
    fig.update_layout(
        title=dict(
            text=f"<b>Monte Carlo Derivation of Euler's Number</b><br>Trials: {n_trials:,} | Result: {estimated_e:.6f}",
            x=0.5,
            font=dict(size=20)
        ),
        template="plotly_dark",
        height=900,
        showlegend=True,
    )
    
    # Axis labels
    fig.update_xaxes(title_text="Dimension ($n$)", row=1, col=1, dtick=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
    fig.update_xaxes(title_text="Number of Trials (Log Scale)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Estimated Value", row=2, col=1, range=[2.5, 3.0]) # Zoom in around e
    
    # Physical/Math Context Annotation
    fig.add_annotation(
        text=r"The dimension $n$ represents how many independent<br>energy variables $X_i \in [0,1)$ are needed<br>for $\sum X_i > 1$.",
        xref="paper", yref="paper",
        x=0.9, y=1.05,
        showarrow=False,
        font=dict(size=12, color="gray"),
        align="right"
    )

    # Output
    output_dir = "output/sec_34"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "euler_simulation.html")
    fig.write_html(output_path)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    trials = 10_000_000
    dims_needed, est_e, n = simulate_euler(n_trials=trials)
    create_visualization(dims_needed, est_e, n)

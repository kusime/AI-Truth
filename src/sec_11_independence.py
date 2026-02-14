import numpy as np
import plotly.graph_objects as go

# --- User's Intuition ---
# "当超过那个临界维度之后...他们其实就是一个静止的点了"
# "Linear Independence = Static points in high dimensions"
# "Understanding: High dimension -> Orthogonality (90 degrees) -> Non-interference"

def measure_independence_and_orthogonality(n_samples=100, max_dim=200):
    dimensions = np.arange(2, max_dim + 1, 2)
    
    avg_angles = []
    matrix_ranks = []
    
    print(f"Running simulation: {n_samples} vectors, Dimensions 2 -> {max_dim}...")
    
    for d in dimensions:
        # 1. Generate Random Gaussian Vectors
        # Gaussian is better for checking orthogonality than Uniform
        X = np.random.randn(n_samples, d)
        
        # 2. Measure "Static-ness" (Orthogonality) as Average Angle
        # We normalize vectors first to use dot product as cosine
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        # Compute pairwise dot products (Cosine similarity)
        # We take a subset for speed
        subset_size = min(n_samples, 100)
        dots = np.dot(X_norm[:subset_size], X_norm[:subset_size].T)
        
        # Avoid self-dot products (diagonal is always 1)
        np.fill_diagonal(dots, np.nan)
        avg_cosine = np.nanmean(np.abs(dots)) # Avg absolute cosine
        
        # Convert cosine to degrees: acos(0) = 90 deg
        # We want to show how close to 90 degrees (0 cosine) we get
        avg_angle = np.degrees(np.arccos(avg_cosine))
        avg_angles.append(avg_angle)
        
        # 3. Measure Linear Independence (Matrix Rank)
        # Rank tells us how many "independent directions" exist
        # If Rank == N, then every point has its own dimension (Static/Independent)
        rank = np.linalg.matrix_rank(X)
        matrix_ranks.append(rank)

        if d % 50 == 2 or d == n_samples:
            print(f"Dim {d}: Rank = {rank}/{n_samples}, Avg Angle = {avg_angle:.2f}°")

    return dimensions, avg_angles, matrix_ranks

def create_plots(dims, angles, ranks, n_samples):
    # Plot 1: Orthogonality (Angle -> 90)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dims, y=angles, mode='lines', name='Avg Angle', line=dict(color='#00f2ff', width=3)))
    fig1.add_hline(y=90, line_dash="dash", line_color="white", annotation_text="Orthogonal (90°)")
    
    fig1.update_layout(
        title="1. The 'Static' Proof: Random Vectors become Orthogonal",
        xaxis_title="Dimension",
        yaxis_title="Average Angle (Degrees)",
        template="plotly_dark",
        yaxis=dict(range=[45, 95])
    )
    
    # Plot 2: Linear Independence (Rank Saturation)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dims, y=ranks, mode='lines', name='Matrix Rank', line=dict(color='#ff0055', width=3)))
    
    # Add critical threshold line
    fig2.add_vline(x=n_samples, line_dash="dot", line_color="yellow", annotation_text="Critical Dimension = N")
    fig2.add_hline(y=n_samples, line_dash="dash", line_color="green", annotation_text="Full Independence")
    
    fig2.update_layout(
        title=f"2. Linear Independence: Critical Phase Transition at Dim={n_samples}",
        xaxis_title="Dimension",
        yaxis_title="Matrix Rank (Independent Vectors)",
        template="plotly_dark"
    )
    
    return fig1, fig2

if __name__ == "__main__":
    N = 100
    DIM_LIMIT = 200
    
    dims, angles, ranks = measure_independence_and_orthogonality(n_samples=N, max_dim=DIM_LIMIT)
    
    fig_angle, fig_rank = create_plots(dims, angles, ranks, N)
    
    fig_angle.write_html("output/sec_11/independence_angle.html")
    fig_rank.write_html("output/sec_11/independence_rank.html")
    
    print("\nVerification Complete.")
    print("1. 'independence_angle.html': Shows vectors approaching 90 degrees (Orthogonality).")
    print("2. 'independence_rank.html': Shows Rank hitting 100 exactly when Dim >= 100.")

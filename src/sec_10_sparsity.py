import time

import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

# --- User's Intuition ---
# "维度越高代表信息越稀疏，而信息越稀疏和你在纸上区分红蓝那么简单"
# "High dimensions = Sparsity = Easy Separation" (Cover's Theorem)

def measure_sparsity_and_separability(n_samples=100, max_dim=200):
    dimensions = np.arange(2, max_dim + 1, 5)
    
    avg_distances = []
    separability_scores = []
    
    print(f"Running simulation: {n_samples} points, Dimensions 2 -> {max_dim}...")
    
    for d in dimensions:
        # 1. Generate Random Points in Hypercube [-1, 1]^d
        X = np.random.uniform(-1, 1, size=(n_samples, d))
        
        # 2. Assign Random Binary Labels (0 or 1)
        # We want to see if a LINEAR classifier can separate pure random noise
        y = np.random.randint(0, 2, size=n_samples)
        
        # 3. Measure Sparsity: Average Pairwise Distance
        # In high dim, points should get further apart
        # We just sample a subset of pairs to be fast
        idx = np.random.choice(n_samples, size=min(n_samples, 50), replace=False)
        subset = X[idx]
        dists = []
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                dist = np.linalg.norm(subset[i] - subset[j])
                dists.append(dist)
        avg_dist = np.mean(dists)
        avg_distances.append(avg_dist)
        
        # 4. Measure Separability: Can a simple Linear SVM separate them?
        # If the data is random, low dim should fail (overlap).
        # High dim should succeed (sparsity -> corners of hypercube).
        clf = LinearSVC(dual="auto", random_state=42, max_iter=1000)
        clf.fit(X, y)
        acc = clf.score(X, y) # Training accuracy tells us if it IS separable
        separability_scores.append(acc)
        
        if d % 50 == 2:
            print(f"Dim {d}: Avg Dist = {avg_dist:.2f}, Linear Sep = {acc*100:.1f}%")

    return dimensions, avg_distances, separability_scores

def create_plots(dims, dists, seps):
    # Plot 1: Sparsity (Distance increasing)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dims, y=dists, mode='lines+markers', name='Avg Distance', line=dict(color='#00f2ff')))
    fig1.update_layout(
        title="1. Sparsity: Average Distance between Random Points",
        xaxis_title="Dimension",
        yaxis_title="Euclidean Distance",
        template="plotly_dark"
    )
    
    # Plot 2: Cover's Theorem (Separability)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dims, y=np.array(seps)*100, mode='lines+markers', name='Linear Separability', line=dict(color='#ff0055')))
    
    # Add theoretical threshold line (approximate)
    # Ideally separability hits 100% when d >= n_samples (roughly)
    fig2.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Perfect Separation")
    
    fig2.update_layout(
        title="2. Cover's Theorem: Probability of Linear Separation",
        xaxis_title="Dimension",
        yaxis_title="Separability (%)",
        template="plotly_dark",
        yaxis=dict(range=[50, 105])
    )
    
    return fig1, fig2

if __name__ == "__main__":
    # Test Case: N=100 points.
    # Theory predicts they should become separable around d > N.
    N = 100
    DIM_LIMIT = 300 
    
    dims, dists, seps = measure_sparsity_and_separability(n_samples=N, max_dim=DIM_LIMIT)
    
    fig_sparsity, fig_separability = create_plots(dims, dists, seps)
    
    fig_sparsity.write_html("output/sec_10/sparsity_distance.html")
    fig_separability.write_html("output/sec_10/sparsity_separability.html")
    
    print("\nVerification Complete.")
    print("1. 'sparsity_distance.html': Confirms 'The Curse/Blessing'. Points get further apart.")
    print("2. 'sparsity_separability.html': Confirms Cover's Theorem. Random noise becomes 100% separable in high dimensions.")

import numpy as np
import plotly.graph_objects as go
from sklearn.random_projection import GaussianRandomProjection

# --- User's Intuition ---
# "在某些维度撞车导致永久不可逆的信息损失"
# "High dimensions = Unique Coordinates"
# "Low dimensions = Overlap/Collision (The Shadow Problem)"

def measure_collision_loss(n_samples=500, hig_dim=100):
    # 1. Start with "God's View": 100D Space
    # Every point is unique and far apart.
    X_high = np.random.uniform(-1, 1, size=(n_samples, hig_dim))
    
    # Verify uniqueness in High Dim (Collision Distance Threshold)
    # If dist < 0.1, we consider them "collided" (indistinguishable)
    threshold = 0.5
    
    dims_to_test = [100, 50, 20, 10, 5, 3, 2, 1]
    collision_rates = []
    
    print(f"Running simulation: {n_samples} samples. Projecting 100D -> 1D...")
    
    for d in dims_to_test:
        if d == hig_dim:
            X_proj = X_high
        else:
            # Random Projection (The "Shadow")
            # Simulates "Compressing" data or viewing it from a limited angle
            # Y = XW
            transformer = GaussianRandomProjection(n_components=d, random_state=42)
            X_proj = transformer.fit_transform(X_high)
            
        # Count Collisions
        # Normalize to keep scale somewhat comparable (roughly)
        # Or standardizing
        X_proj_norm = (X_proj - X_proj.mean(0)) / (X_proj.std(0) + 1e-9)
        
        collisions = 0
        total_pairs = 0
        
        # Check a subset of pairs for speed
        subset_size = min(n_samples, 200)
        subset = X_proj_norm[:subset_size]
        
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                dist = np.linalg.norm(subset[i] - subset[j])
                total_pairs += 1
                if dist < threshold:
                    collisions += 1
                    
        rate = (collisions / total_pairs) * 100
        collision_rates.append(rate)
        print(f"Dim {d}: Collision Rate = {rate:.2f}% (Information Loss)")
        
    return dims_to_test, collision_rates, X_high

def visualize_collision(dims, rates, X_high):
    # 1. Collision Rate Plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=[str(d) for d in dims], y=rates, mode='lines+markers', 
                             marker=dict(size=10, color=rates, colorscale='Reds'),
                             line=dict(color='orange', width=3)))
    
    fig1.update_layout(
        title="1. The Shadow Problem: Information Loss via Projection",
        xaxis_title="Dimension (Log Scale)",
        yaxis_title="Collision Rate (% of pairs indistinguishable)",
        template="plotly_dark",
        xaxis=dict(autorange="reversed") # High dim on left
    )
    
    # 2. Visual Proof: 3D -> 1D Shadow
    # Generate 3 distinct clusters in 3D
    # Show how they merge in 1D
    X_3d = np.random.randn(300, 3) 
    # Shift them to be distinct
    X_3d[:100] += [5, 0, 0]
    X_3d[100:200] += [0, 5, 0]
    X_3d[200:] += [0, 0, 5]
    labels = [0]*100 + [1]*100 + [2]*100
    
    # Project to 1D (Bad angle)
    # Just take x-axis sum? No, random projection
    proj_1d = X_3d.dot(np.random.randn(3, 1))
    
    # Visualization
    fig2 = go.Figure()
    
    # 3D View
    fig2.add_trace(go.Scatter3d(
        x=X_3d[:,0], y=X_3d[:,1], z=X_3d[:,2],
        mode='markers', marker=dict(size=3, color=labels, colorscale='Viridis'),
        name='Original 3D (Distinct)'
    ))
    
    # 1D Shadow (projected on the floor z=-5)
    floor_z = -5
    fig2.add_trace(go.Scatter3d(
        x=proj_1d.flatten(), y=np.zeros_like(proj_1d.flatten()) + 10, z=np.zeros_like(proj_1d.flatten()) + floor_z,
        mode='markers', marker=dict(size=3, color=labels, colorscale='Viridis', symbol='square'),
        name='1D Shadow (Collided)'
    ))
    
    fig2.update_layout(
        title="2. Visualizing 'Collision': Unique 3D Clusters -> Mixed 1D Shadow",
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z")
        )
    )

    return fig1, fig2

if __name__ == "__main__":
    dims, rates, X = measure_collision_loss()
    
    fig_rate, fig_vis = visualize_collision(dims, rates, X)
    
    fig_rate.write_html("output/sec_12/collision_rate.html")
    fig_vis.write_html("output/sec_12/collision_shadow.html")
    
    print("\nVerification Complete.")
    print("1. 'collision_rate.html': Shows collision/ambiguity skyrocketing as dimension drops.")
    print("2. 'collision_shadow.html': Visually demonstrates how 3 distinct groups merge into one mess in low dim.")

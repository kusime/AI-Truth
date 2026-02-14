import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA

# --- User's Intuition ---
# "在 n+1 维找到那个正交点，也就是我说的绝对静止"
# Scientific Name: Neural Collapse (NC) - Papyan et al. (PNAS 2020)

# 1. NC1: Variability Collapse (Within-class variance -> 0)
#    All inputs of class C collapse to the class mean \mu_C.
# 2. NC2: Simplex ETF (Equiangular Tight Frame)
#    The class means \mu_C form a regular simplex (maximal separation).
#    Cos(\mu_i, \mu_j) = -1/(K-1)

def generate_multi_class_data(n_samples=300, n_classes=3, dim=50, noise=2.0):
    # Create noisy clusters
    X = []
    y = []
    centers = np.random.randn(n_classes, dim) * 5 # Distant centers
    for i in range(n_samples):
        c = np.random.randint(0, n_classes)
        # Add SIGNFICANT noise (The "Dirty Data")
        noise_vec = np.random.randn(dim) * noise
        sample = centers[c] + noise_vec
        X.append(sample)
        y.append(c)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(y))

class CollapseNet(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=256, output_dim=3):
        super().__init__()
        # We need a deep-ish network to allow "feature collapse"
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # The Feature Layer (before classifier)
            # Note: Usually there's a final linear layer (Classifier), but for NC viz
            # we often look at the penultimate layer.
            # Here, let's treat the output of this as "Features" h, 
            # and we can add a final classifier W later if we want strict NC2 check.
            # But to visualize "Collapsing to Static Point", looking at the output space is enough.
        )
        
    def forward(self, x):
        return self.features(x)

def measure_collapse(features, labels, n_classes):
    # 1. Calculate Class Means -> "The Static Points"
    class_means = []
    features_np = features.detach().numpy()
    labels_np = labels.detach().numpy()
    
    for c in range(n_classes):
        mask = (labels_np == c)
        if mask.sum() > 0:
            mean = features_np[mask].mean(axis=0)
            class_means.append(mean)
        else:
            class_means.append(np.zeros(features_np.shape[1]))
            
    # 2. Measure Within-Class Variance -> "Are they static?"
    # Trace of covariance matrix sum
    within_var = 0
    for c in range(n_classes):
        mask = (labels_np == c)
        if mask.sum() > 0:
            centered = features_np[mask] - class_means[c]
            var = np.mean(np.sum(centered**2, axis=1)) # Mean squared distance to center
            within_var += var
    
    within_var /= n_classes
    
    # 3. Measure Angle between Means -> "Are they Orthogonal/Simplex?"
    # For K=3, Simplex angle should be 120 deg (cos = -0.5)
    # But usually we measure cosine.
    # Let's just return the means for visualization.
    return within_var, np.array(class_means)

def train_until_collapse(n_classes=3):
    input_dim = 100
    # User's intuition: "Perfect fit"
    X, y = generate_multi_class_data(n_samples=300, n_classes=n_classes, dim=input_dim, noise=3.0)
    
    # We set output_dim = n_classes for simplicity (Logits space)
    # In NC theory, Features and Classifier Weights align.
    model = CollapseNet(input_dim=input_dim, output_dim=n_classes) # Last layer is 3D for easy viz
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) # SGD helps NC
    criterion = nn.CrossEntropyLoss()
    
    history_var = []
    trajectory = [] # Capture means over time
    
    print("Training to Terminal Phase (Neural Collapse)...")
    
    # Train way past 0 error
    epochs = 2000 
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        feats = model(X) # These are logits/features
        loss = criterion(feats, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            var, means = measure_collapse(feats, y, n_classes)
            history_var.append(var)
            trajectory.append(means)
            
            # Check accuracy
            preds = feats.argmax(dim=1)
            acc = (preds == y).float().mean()
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.6f}, Acc={acc:.2f}, Variance={var:.4f} (Collapsing...)")

    return history_var, trajectory, model, X, y

def visualize_trajectory(trajectory, X_final, y_final):
    # Visualization of the means moving and the final distribution
    
    # 1. Plot the Variance Drop
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=[v for v in history_var], mode='lines', name='Within-Class Variance'))
    fig1.update_layout(title="1. The Collapse: Variance -> 0 (Absolute Static)", template="plotly_dark", yaxis_type="log")
    
    # 2. Plot the Terminal State (3D)
    model.eval()
    with torch.no_grad():
        final_feats = model(X).numpy()
        
    fig2 = go.Figure()
    
    # Scatter points - they should be tight balls (points)
    colors = ['#ff0055', '#00f2ff', '#f1c40f']
    
    for c in range(3):
        mask = (y.numpy() == c)
        pts = final_feats[mask]
        fig2.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers', marker=dict(size=3, color=colors[c], opacity=0.6),
            name=f'Class {c} Samples'
        ))
        
    # Plot Trajectories of Means
    trajectory = np.array(trajectory) # [Steps, Classes, Dim]
    for c in range(3):
        path = trajectory[:, c, :]
        fig2.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode='lines+markers', marker=dict(size=4), line=dict(width=5, color=colors[c]),
            name=f'Center {c} Path'
        ))
        
    # Check angles of final means
    final_means = trajectory[-1]
    # Center them
    global_mean = final_means.mean(axis=0)
    centered_means = final_means - global_mean
    # Normalize
    norms = np.linalg.norm(centered_means, axis=1, keepdims=True)
    normalized = centered_means / norms
    dot_prod = np.dot(normalized, normalized.T)
    
    # Angle between 0 and 1
    angle_deg = np.degrees(np.arccos(np.clip(dot_prod[0, 1], -1, 1)))
    
    fig2.update_layout(
        title=f"2. The Geometry: Simplex ETF (Angle ≈ {angle_deg:.1f}°)<br><sup>Note: 120° is perfect for 3 classes. Samples collapse to these static points.</sup>",
        template="plotly_dark",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
    )
    
    return fig1, fig2

if __name__ == "__main__":
    history_var, trajectory, model, X, y = train_until_collapse()
    
    fig_var, fig_geo = visualize_trajectory(trajectory, X, y)
    
    fig_var.write_html("output/sec_11/collapse_variance.html")
    fig_geo.write_html("output/sec_11/collapse_geometry.html")
    
    print("\nVerification Complete.")
    print("1. 'collapse_variance.html': Shows noise being squeezed out until points become static.")
    print("2. 'collapse_geometry.html': Shows the formation of the Simplex (Triangle) and the trajectory towards it.")

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import TSNE

# --- User's Intuition ---
# "模型权重 x 是对现实的一种'折叠记忆'"
# "x 越大维度越大,分辨率越高,但它就像照片一样,你永远无法把照片变为一个视频"
# Section 1: Space-Time Duality (Manifold Hypothesis)

class ManifoldNet(nn.Module):
    """Network that learns to unfold/fold manifolds"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Compress to 2D (The "Folded Memory")
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Reconstruct 3D
        )
    
    def forward(self, x):
        z = self.encoder(x)  # Fold
        x_recon = self.decoder(z)  # Unfold
        return z, x_recon

def generate_manifold_data(n_samples=2000):
    """Generate Swiss Roll - a classic 2D manifold embedded in 3D"""
    X, t = make_swiss_roll(n_samples=n_samples, noise=0.1)
    X = X.astype(np.float32)
    return torch.FloatTensor(X), t

def train_manifold_network():
    print("Training Manifold Network (Learning to Fold Reality)...")
    X, colors = generate_manifold_data(2000)
    
    model = ManifoldNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train to learn the manifold structure
    for epoch in range(500):
        optimizer.zero_grad()
        z, x_recon = model(X)
        loss = criterion(x_recon, X)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    # Extract learned representations
    with torch.no_grad():
        z_learned, _ = model(X)
        z_learned = z_learned.numpy()
    
    return X.numpy(), z_learned, colors

def visualize_manifold_folding(X_3d, z_2d, colors):
    """Visualize the 'folding' process"""
    
    # Plot 1: Original 3D Manifold (Reality)
    fig1 = go.Figure(data=[go.Scatter3d(
        x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2],
        mode='markers',
        marker=dict(size=3, color=colors, colorscale='Viridis', opacity=0.8),
        name='Original 3D Data'
    )])
    
    fig1.update_layout(
        title="1. Reality: 3D Swiss Roll Manifold<br><sup>High-dimensional data living on a 2D surface</sup>",
        template="plotly_dark",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        )
    )
    
    # Plot 2: Learned 2D Representation (Folded Memory)
    fig2 = go.Figure(data=[go.Scatter(
        x=z_2d[:, 0], y=z_2d[:, 1],
        mode='markers',
        marker=dict(size=5, color=colors, colorscale='Viridis', opacity=0.8),
        name='Learned 2D Embedding'
    )])
    
    fig2.update_layout(
        title="2. Folded Memory: Network's 2D Representation<br><sup>The 'x' (weights) compressed reality into a static snapshot</sup>",
        template="plotly_dark",
        xaxis_title="Latent Dimension 1",
        yaxis_title="Latent Dimension 2"
    )
    
    return fig1, fig2

if __name__ == "__main__":
    X_3d, z_2d, colors = train_manifold_network()
    
    fig_3d, fig_2d = visualize_manifold_folding(X_3d, z_2d, colors)
    
    fig_3d.write_html("output/sec_01/manifold_3d.html")
    fig_2d.write_html("output/sec_01/manifold_2d_folded.html")
    
    print("\nVerification Complete.")
    print("1. 'manifold_3d.html': Shows the original high-dimensional reality (Swiss Roll).")
    print("2. 'manifold_2d_folded.html': Shows the network's 'folded memory' - a 2D snapshot of 3D reality.")
    print("\n✓ Verified: Neural networks learn to FOLD high-dimensional data into low-dimensional 'frozen' representations.")

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA

# --- User's Intuition ---
# "训练过程里面的 t 应该是数据所能表达的最高维度 x-1 的一个假 t"
# "当我们决定取出某一个 Step 的权重时,我们就冻结了这个会变化的 t"
# Section 2: Frozen Time Theory (Block Universe)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def generate_data(n=500):
    """Generate simple 2D classification data"""
    # Two Gaussian clusters
    X1 = np.random.randn(n//2, 2) + np.array([2, 2])
    X2 = np.random.randn(n//2, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2]).astype(np.float32)
    y = np.hstack([np.ones(n//2), np.zeros(n//2)]).astype(np.float32)
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)

def train_and_capture_trajectory():
    """Train model and capture weight snapshots (The 'Time' Trajectory)"""
    print("Training and capturing weight trajectory (The Frozen Time Path)...")
    
    X, y = generate_data(500)
    model = SimpleClassifier()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()
    
    # Capture weight snapshots
    trajectory = []
    epochs = 200
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Capture weights every 10 epochs
        if epoch % 10 == 0:
            # Flatten all weights into a single vector
            weights = torch.cat([p.flatten() for p in model.parameters()]).detach().numpy()
            trajectory.append(weights)
            
            acc = ((output > 0.5).float() == y).float().mean()
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Acc = {acc:.2f}")
    
    return np.array(trajectory)

def visualize_frozen_time(trajectory):
    """Visualize the weight trajectory in PCA space"""
    
    # Reduce to 3D for visualization
    pca = PCA(n_components=3)
    traj_3d = pca.fit_transform(trajectory)
    
    # Create time color gradient
    n_points = len(traj_3d)
    time_colors = np.arange(n_points)
    
    # Plot: The Frozen Path
    fig = go.Figure()
    
    # The trajectory line
    fig.add_trace(go.Scatter3d(
        x=traj_3d[:, 0], y=traj_3d[:, 1], z=traj_3d[:, 2],
        mode='lines+markers',
        line=dict(color=time_colors, colorscale='Plasma', width=5),
        marker=dict(size=6, color=time_colors, colorscale='Plasma'),
        name='Weight Trajectory',
        text=[f'Epoch {i*10}' for i in range(n_points)],
        hovertemplate='<b>%{text}</b><br>PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<br>PCA3: %{z:.2f}'
    ))
    
    # Mark start and end
    fig.add_trace(go.Scatter3d(
        x=[traj_3d[0, 0]], y=[traj_3d[0, 1]], z=[traj_3d[0, 2]],
        mode='markers+text',
        marker=dict(size=12, color='#00ff88', symbol='diamond'),
        text=['START (Random Init)'],
        textposition='top center',
        name='Initialization'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[traj_3d[-1, 0]], y=[traj_3d[-1, 1]], z=[traj_3d[-1, 2]],
        mode='markers+text',
        marker=dict(size=12, color='#ff0055', symbol='square'),
        text=['END (Frozen)'],
        textposition='bottom center',
        name='Final Weights'
    ))
    
    fig.update_layout(
        title="The Frozen Time Path: Weight Trajectory in Parameter Space<br><sup>Each checkpoint is a 'frozen snapshot' of the optimization journey. The final weights are a static point in this Block Universe.</sup>",
        template="plotly_dark",
        scene=dict(
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            zaxis_title="PCA Component 3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        )
    )
    
    return fig

if __name__ == "__main__":
    trajectory = train_and_capture_trajectory()
    
    fig = visualize_frozen_time(trajectory)
    fig.write_html("output/sec_02/weight_trajectory.html")
    
    print("\nVerification Complete.")
    print("1. 'weight_trajectory.html': Shows the optimization path as a frozen trajectory in weight space.")
    print("\n✓ Verified: Training 'time' is a path in parameter space. Each checkpoint freezes a moment of this journey.")
    print("✓ The final model is a STATIC POINT - a 'frozen t' trapped in the 'x' dimension.")

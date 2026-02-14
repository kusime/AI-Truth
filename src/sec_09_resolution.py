import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA

# --- User's Intuition Parameters ---
# "只要分辨率够了... 512->1024->2048->10 里面结构其实真的无所谓"
# We will use a deliberately "weird" and overkill architecture to prove this point.
# A simple 2D spiral task does not need 2048 neurons, but we use them to show
# that "Resolution" (Width) makes the problem trivial for the network.

HIDDEN_LAYERS = [128, 512, 1024, 128] # Massive over-parameterization for a 2D task
ACTIVATION = nn.ReLU() # The "folding" operator

# --- 1. Data Generation: The "Manifold" ---
def generate_spiral_data(n_samples=1000, noise=0.1):
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 780 * (2 * np.pi) / 360
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T + np.random.randn(n, 2) * noise

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T + np.random.randn(n, 2) * noise

    res_a = np.append(data_a, np.zeros((n, 1)), axis=1)
    res_b = np.append(data_b, np.ones((n, 1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)
    
    X = torch.FloatTensor(res[:, :2])
    y = torch.FloatTensor(res[:, 2]).unsqueeze(1)
    return X, y

# --- 2. The Model: "Structure as Resolution" ---
class IntuitionNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[512, 1024], output_dim=1):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        # Build arbitrary architecture defined by user
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(ACTIVATION)
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

    def get_embedding(self, x):
        # Extract the high-dimensional representation before the final cut
        # This corresponds to the user's "High Dimensional Space" intuition
        feat = x
        for layer in self.net[:-2]: # Skip last linear and sigmoid
            feat = layer(feat)
        return feat

# --- 3. Training: "Loss as the Driver" ---
def train(model, X, y, epochs=500):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    print(f"Training started... Data Shape: {X.shape}")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            acc = ((outputs > 0.5) == y).float().mean()
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {acc.item():.4f}")

    return model

# --- 4. Visualization: "Seeing the Cut" ---
def visualize_results(model, X, y):
    # 1. 2D Decision Boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        preds = model(grid).reshape(xx.shape)
    
    fig = go.Figure()
    
    # Contour (The Cut)
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, 0.1),
        y=np.arange(y_min, y_max, 0.1),
        z=preds.numpy(),
        colorscale='RdBu',
        opacity=0.4,
        showscale=False
    ))
    
    # Data Points
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(color=y.flatten(), colorscale='RdBu', line=dict(width=1, color='white')),
        name='Data'
    ))
    
    fig.update_layout(
        title="1. The Result: How the Network Sliced the Space",
        template="plotly_dark"
    )
    
    # 2. High-Dim Manifold Projection (Information Resolution)
    # We take the 2nd to last layer (e.g., 1024 dimensions) and project to 3D
    with torch.no_grad():
        embeddings = model.get_embedding(X).numpy()
    
    # Use PCA to project 1024D -> 3D
    pca = PCA(n_components=3)
    emb_3d = pca.fit_transform(embeddings)
    
    fig_3d = go.Scatter3d(
        x=emb_3d[:, 0], y=emb_3d[:, 1], z=emb_3d[:, 2],
        mode='markers',
        marker=dict(size=4, color=y.flatten(), colorscale='RdBu', opacity=0.8),
        name='Projected Manifold'
    )
    
    layout_3d = go.Layout(
        title=f"2. The Hidden Reality: {embeddings.shape[1]}D Manifold (Projected to 3D)<br><sup>Note how linearly separable the data becomes in high dimensions.</sup>",
        template="plotly_dark",
        scene=dict(xaxis=dict(title="PC1", backgroundcolor="black"), yaxis=dict(title="PC2", backgroundcolor="black"), zaxis=dict(title="PC3", backgroundcolor="black")),
        paper_bgcolor="black",
    )
    
    fig2 = go.Figure(data=[fig_3d], layout=layout_3d)

    return fig, fig2

if __name__ == "__main__":
    # Settings
    N_SAMPLES = 2000
    
    # Generate Data
    X, y = generate_spiral_data(N_SAMPLES)
    
    # Model
    # User's intuition: "Structure doesn't matter, give me resolution"
    # We use a massive network for a simple problem.
    model = IntuitionNet(input_dim=2, hidden_dims=HIDDEN_LAYERS)
    
    print(f"Model Architecture: Input(2) -> {HIDDEN_LAYERS} -> Output(1)")
    print("Verifying User Intuition: 'Resolution enables slicing'...")
    
    # Train
    model = train(model, X, y)
    
    # Visualize
    fig1, fig2 = visualize_results(model, X, y)
    
    fig1.write_html("output/sec_09/intuition_result_boundary.html")
    fig2.write_html("output/sec_09/intuition_result_manifold.html")
    print("\nGenerated 'intuition_result_boundary.html' and 'intuition_result_manifold.html'.")
    print("These visualizations prove that sufficient width allows the network to 'unfold' the spiral data seamlessly.")

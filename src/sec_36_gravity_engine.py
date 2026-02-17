
import gzip
import os
import struct
import time
import urllib.request

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration ---
DATA_DIR = "data/mnist"
OUTPUT_DIR = "output/sec_36"
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz"
}

# ==========================================
# PART 1: The Infrastructure (Data Loader)
# ==========================================

def download_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    for key, filename in FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            url = MNIST_URL + filename
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                # Fallback to LeCun's site if mirror fails
                alt_url = "http://yann.lecun.com/exdb/mnist/" + filename
                print(f"Trying alternative {alt_url}...")
                urllib.request.urlretrieve(alt_url, filepath)
    print("Data download check complete.")

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {filename}")
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num, rows * cols)
        return data.astype(np.float32) / 255.0 # Normalize 0-1

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {filename}")
        buffer = f.read()
        return np.frombuffer(buffer, dtype=np.uint8)

def get_data():
    download_mnist()
    train_x = load_images(os.path.join(DATA_DIR, FILES['train_img']))
    train_y = load_labels(os.path.join(DATA_DIR, FILES['train_label']))
    test_x = load_images(os.path.join(DATA_DIR, FILES['test_img']))
    test_y = load_labels(os.path.join(DATA_DIR, FILES['test_label']))
    return train_x, train_y, test_x, test_y

# ==========================================
# PART 2: The Physics Engine (Gravity Model)
# ==========================================

class GravityEngine:
    """
    A Physical Classifier based on Gravitational Potential Energy.
    Each Digit Class (0-9) creates a 'Gravity Well' in 784D space.
    The Potential is defined by the mass distribution (Mean & Variance).
    """
    def __init__(self):
        self.wells = []
        
    def fit(self, X, y):
        print("Constructing Gravity Wells (Training)...")
        self.wells = []
        for digit in range(10):
            # 1. Filter Mass for this digit
            mass_block = X[y == digit]
            
            # 2. Calculate Center of Mass (Mean)
            center = np.mean(mass_block, axis=0)
            
            # 3. Calculate Field Spread (Variance)
            # Var represents 'Temperature' or 'Entropy' of the mass
            # Add epsilon to prevent infinite gravity (division by zero)
            variance = np.var(mass_block, axis=0) + 1e-3
            
            self.wells.append({
                'label': digit,
                'mu': center,
                'sigma2': variance,
                # Precompute log term for speed: 0.5 * sum(log(2*pi*sigma^2))
                'entropy_term': 0.5 * np.sum(np.log(2 * np.pi * variance))
            })
            
    def compute_energy(self, x_particle, well):
        """
        Calculates Potential Energy of a particle x in a specific gravity well.
        E = Kinetic/Distance Term + Entropy Term
        E = 0.5 * sum( (x - mu)^2 / sigma^2 ) + sum(log(sigma))
        """
        # x_particle: (784,)
        # well['mu']: (784,)
        
        # Distance squared, weighted by field strength (1/variance)
        diff_sq = (x_particle - well['mu'])**2
        distance_energy = 0.5 * np.sum(diff_sq / well['sigma2'])
        
        # Total Energy (Negative Log Likelihood)
        total_energy = distance_energy + well['entropy_term']
        return total_energy

    def predict(self, X_test):
        print("Simulating Particle Trajectories (Predicting)...")
        predictions = []
        energies = []
        
        # Vectorized implementation for speed could be done, but loop clarifies physics
        # To speed up 10k items, we'll try to batch.
        
        # Matrix form: 
        # dist_sq[N, 784] = (X[N] - mu)^2
        # E[N] = sum(dist_sq / sig2, axis=1)
        
        N = X_test.shape[0]
        results = np.zeros((N, 10)) # Energy matrix
        
        for k in range(10):
            well = self.wells[k]
            # (X - mu)^2
            diff = X_test - well['mu']
            diff_sq = diff**2
            
            # Sum over dimensions
            # 0.5 * sum( (x-mu)^2 / sig2 )
            dist_term = 0.5 * np.sum(diff_sq / well['sigma2'], axis=1)
            
            total_e = dist_term + well['entropy_term']
            results[:, k] = total_e
            
        # Prediction: The well with MINIMUM Potential Energy
        predictions = np.argmin(results, axis=1)
        return predictions, results

# ==========================================
# PART 3: Visualization & Verification
# ==========================================

def visualize_gravity_slice(engine, digit_a=3, digit_b=8):
    """
    Visualizes the Gravitational Potential along a path between two digits.
    Shows the 'Energy Barrier' and the 'Wells'.
    """
    print(f"Generating Gravity Slice between {digit_a} and {digit_b}...")
    
    well_a = engine.wells[digit_a]
    well_b = engine.wells[digit_b]
    
    # Linear path in 784D space
    alpha = np.linspace(-0.5, 1.5, 100)
    
    # Energies along the path
    energies_a = []
    energies_b = []
    
    for a in alpha:
        # Physical location x(a)
        x_loc = well_a['mu'] * (1-a) + well_b['mu'] * a
        
        # Calculate potential w.r.t well A and well B
        e_a = engine.compute_energy(x_loc, well_a)
        e_b = engine.compute_energy(x_loc, well_b)
        
        energies_a.append(e_a)
        energies_b.append(e_b)
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=alpha, y=energies_a,
        mode='lines', name=f'Potential of "{digit_a}"',
        line=dict(color='#00f2ff', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=alpha, y=energies_b,
        mode='lines', name=f'Potential of "{digit_b}"',
        line=dict(color='#ff0055', width=3)
    ))
    
    # Formatting
    fig.update_layout(
        title=f"<b>Gravitational Potential Slice ({digit_a} vs {digit_b})</b><br>Path from Center {digit_a} (x=0) to Center {digit_b} (x=1)",
        xaxis_title="Path Coordinates (Interpolation)",
        yaxis_title="Potential Energy (Log Loss)",
        template="plotly_dark",
        height=600
    )
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mnist_gravity_slice.html")
    fig.write_html(out_path)
    print(f"Slice Viz saved to {out_path}")


def visualize_gravity_clouds(engine):
    """
    Visualizes the "Collapsed Core" (Mean) and "Event Horizon" (Variance) 
    of the high-dimensional gravity clouds.
    """
    print("Collapsing Gravity Clouds into Observable Images...")
    
    # We want 2 rows of 10 digits
    # Row 1: The Singularity (Mean) - The 'Ideal' Digit
    # Row 2: The Event Horizon (Variance) - The 'Uncertainty' Halo
    
    fig = make_subplots(
        rows=2, cols=10,
        subplot_titles=[f"Singularity {i}" for i in range(10)] + [f"Horizon {i}" for i in range(10)],
        horizontal_spacing=0.02,
        vertical_spacing=0.1
    )
    
    for digit in range(10):
        well = engine.wells[digit]
        
        # 1. Reconstruct the "Core Mass"
        # This is the 784D mean vector reshaped back to 2D
        core_img = well['mu'].reshape(28, 28)
        
        # 2. Reconstruct the "Field Variance"
        # High variance = Low mass density (Fuzzy edges)
        # Low variance = High mass density (Solid stroke)
        # We visualize log(variance) to see the range better
        var_img = np.log(well['sigma2'].reshape(28, 28))
        
        # Add Core Trace (Heatmap)
        fig.add_trace(go.Heatmap(
            z=core_img,
            colorscale='Viridis',
            showscale=False,
            name=f'Core {digit}'
        ), row=1, col=digit+1)
        
        # Add Variance Trace (Heatmap)
        # Invert colorscale so dark = low variance (solid)
        fig.add_trace(go.Heatmap(
            z=var_img,
            colorscale='Magma', 
            showscale=False,
            name=f'Horizon {digit}'
        ), row=2, col=digit+1)
        
        # Fix aspect ratio
        fig.update_yaxes(scaleanchor=f"x{digit+1}", row=1, col=digit+1, visible=False, autorange='reversed')
        fig.update_xaxes(visible=False, row=1, col=digit+1)
        fig.update_yaxes(scaleanchor=f"x{digit+11}", row=2, col=digit+1, visible=False, autorange='reversed')
        fig.update_xaxes(visible=False, row=2, col=digit+1)

    fig.update_layout(
        title="<b>Gravity Clouds Collapsed</b><br>Top: The 'Ideal' Mass Center. Bottom: The Quantum Uncertainty Field (Low Variance = High Structure)",
        height=400,
        margin=dict(l=10, r=10, t=80, b=10),
        template="plotly_dark"
    )
    
    out_path = os.path.join(OUTPUT_DIR, "mnist_gravity_clouds.html")
    fig.write_html(out_path)
    print(f"Gravity Clouds Viz saved to {out_path}")

def main():
    start_time = time.time()
    
    # 1. Load Data
    try:
        train_x, train_y, test_x, test_y = get_data()
        print(f"Data Loaded: Train {train_x.shape}, Test {test_x.shape}")
    except Exception as e:
        print(f"FATAL: Failed to get MNIST data. {e}")
        return

    # 2. Train Physics Engine
    engine = GravityEngine()
    engine.fit(train_x, train_y)
    
    # 3. Predict / Simulate
    preds, energy_matrix = engine.predict(test_x)
    
    # 4. Verify Accuracy
    acc = np.mean(preds == test_y)
    print("="*40)
    print(f"PHYSICS ENGINE PROOF COMPLETE")
    print(f"Gravitational Accuracy: {acc*100:.2f}%")
    print("="*40)
    
    # 5. Visualize
    visualize_gravity_slice(engine, 3, 8)
    visualize_gravity_clouds(engine)
    
    print(f"Total Proof Time: {time.time() - start_time:.2f}s")
    
if __name__ == "__main__":
    main()

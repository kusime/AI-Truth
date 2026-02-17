
import os
import sys
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path to import GravityEngine
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.sec_36_gravity_engine import GravityEngine, get_data

# --- Configuration ---
OUTPUT_DIR = "output/sec_38"
NOISE_INTENSITY = 0.8  # How "bright" the noise is (0-1)
SEQUENCE_LENGTH = 56   # 28 rows * 2 (Dilated by 2x for noise injection)

# ==========================================
# PART 1: The Physics of "Selection"
# ==========================================

class MassSelector:
    """
    The Physics Engine's 'Gatekeeper'.
    Calculates the 'Gravitational Mass' (Importance) of an incoming data packet.
    """
    @staticmethod
    def calculate_mass(frame):
        """
        Input: frame (28,) - A single row of pixels.
        Output: mass (scalar 0-1)
        
        Physics Heuristic V2: "Smoothness implies Structure" (Total Variation).
        - Real Strokes: [0 0 1 1 1 1 0 0]. Jumps = 2. Low TV.
        - Noise: [0.2 0.8 0.1 0.9 ...]. Jumps = Many. High TV.
        """
        max_val = np.max(frame)
        
        # Rule A: Silence has no mass
        if max_val < 0.2:
            return 0.0
            
        # Rule B: Calculate Total Variation (sum of absolute differences)
        # Measures "High Frequency" content
        diffs = np.abs(np.diff(frame))
        tv = np.sum(diffs)
        
        # Theoretical Max TV for 0-1 noise: ~14.0 (avg jump 0.5 * 28)
        # Theoretical TV for stroke: ~2.0 (up and down)
        
        # Threshold: If it wiggles too much, it's Entropy.
        if tv > 4.5: 
            return 0.0 # Reject Entropy
            
        # Rule C: It's a clean, smooth structure (Signal)
        return 1.0

class GravityMamba:
    """
    Selective Gravity Accumulator.
    Simulates: h_t = h_{t-1} + Mass_t * (x_t - h_{t-1})
    """
    def __init__(self, shape=(28, 28)):
        self.shape = shape
        self.h = np.zeros(shape) # The "Hidden State" (Canvas)
        self.history = []
        self.mass_history = []
        
    def step(self, frame, row_idx):
        """
        Absorb a row into the state, BUT only if it has Mass.
        row_idx: Where this frame *belongs* spatially (if we know it).
        In a pure RNN, we might not know row_idx, but for Image Reconstruction,
        we assume we are 'scanning' the image.
        """
        # 1. Calculate Mass (Selection)
        mass = MassSelector.calculate_mass(frame)
        self.mass_history.append(mass)
        
        # 2. Update State
        # We only update the specific row in the state (Conceptually 'Writing' to memory)
        # OR: We could update the whole state vector if it was holographic.
        # For visual clarity: We write to the specific row of the image canvas.
        
        current_row_val = self.h[row_idx, :]
        
        # Physics Update: Conservation of Momentum?
        # New = Old + Mass * (Input - Old) -> Exponential Moving Average-ish
        # If Mass=1, New = Input (Full Replace)
        # If Mass=0, New = Old (Ignore Input)
        
        new_row_val = current_row_val * (1 - mass) + frame * mass
        self.h[row_idx, :] = new_row_val
        
        self.history.append(self.h.copy())

class NaiveAccumulator:
    """
    The 'Transformer' / Standard RNN.
    Absorbs EVERYTHING. No Gating.
    """
    def __init__(self, shape=(28, 28)):
        self.shape = shape
        self.h = np.zeros(shape)
        self.history = []
        
    def step(self, frame, row_idx):
        # Naive: Mass is always 1.0 (or fixed learning rate)
        # We just overwrite/add. 
        # If we overwrite, noise overwrites signal.
        # Let's say we 'Add' to the canvas? No, that overflows.
        # Let's say we 'Replace' based on time.
        
        # If multiple frames map to same row (noise injection), 
        # the latest one wins in a naive system without history protection.
        self.h[row_idx, :] = frame 
        self.history.append(self.h.copy())


# ==========================================
# PART 2: The Stream Generation
# ==========================================

def create_noisy_stream(digit_img, noise_level=0.5):
    """
    Takes a 28x28 digit.
    Returns a stream of (row_idx, frame_pixels).
    Injects random noise frames in between real frames.
    """
    stream = []
    
    # Iterate through real rows
    for r in range(28):
        # 1. The Real Signal
        real_row = digit_img[r, :]
        stream.append({
            'type': 'signal',
            'row_idx': r,
            'data': real_row
        })
        
        # 2. The Entropy Injection (Time Travel Garbage)
        # Random chance to insert noise packets
        if np.random.rand() < 0.8: # 80% chance of noise between lines
            noise_row = np.random.rand(28) * noise_level 
            # Make it "White Noise" (High Entropy)
            
            # Target a random row index to mess up the state
            # (Simulating "Future" or "Past" confusion)
            target_r = r 
            
            stream.append({
                'type': 'noise',
                'row_idx': target_r,
                'data': noise_row
            })
            
    return stream

# ==========================================
# PART 3: Visualization & Comparison
# ==========================================

def visualize_mamba_physics(stream, mamba_hist, naive_hist, mass_log, digit_label):
    """
    HTML Animation of the process.
    """
    print("Generating Animation...")
    
    steps = len(stream)
    
    # Create Figure
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Incoming Stream (Time)", "Naive Integration (Transformer)", "Mamba Physics (Selection)",
            "Signal Mass (Selection Gate)", "Result: Naive", "Result: Mamba"
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "scatter", "colspan": 1}, {"type": "heatmap"}, {"type": "heatmap"}] 
        ],
        vertical_spacing=0.15
    )
    
    # Add initial traces
    # 1. Incoming Stream (1D strip)
    fig.add_trace(go.Heatmap(z=np.zeros((1, 28)), colorscale='Greys', zmin=0, zmax=1), row=1, col=1)
    
    # 2. Naive State
    fig.add_trace(go.Heatmap(z=np.zeros((28, 28)), colorscale='Greys', zmin=0, zmax=1), row=1, col=2)
    
    # 3. Mamba State
    fig.add_trace(go.Heatmap(z=np.zeros((28, 28)), colorscale='Greys', zmin=0, zmax=1), row=1, col=3)
    
    # 4. Mass Log (Scatter)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', line=dict(color='cyan')), row=2, col=1)
    
    # 5. Final Results (Placeholders, updated in frames)
    fig.add_trace(go.Heatmap(z=np.zeros((28, 28)), colorscale='Greys', zmin=0, zmax=1), row=2, col=2)
    fig.add_trace(go.Heatmap(z=np.zeros((28, 28)), colorscale='Greys', zmin=0, zmax=1), row=2, col=3)

    # Frames
    frames = []
    
    for t in range(steps):
        packet = stream[t]
        row_idx = packet['row_idx']
        data = packet['data']
        is_noise = packet['type'] == 'noise'
        
        # Current Mass
        mass = mass_log[t]
        
        # Viz: Incoming row as a 1x28 bar
        incoming_viz = data.reshape(1, 28)
        
        frames.append(go.Frame(
            data=[
                go.Heatmap(z=incoming_viz), # Stream
                go.Heatmap(z=naive_hist[t]), # Naive
                go.Heatmap(z=mamba_hist[t]), # Mamba
                go.Scatter(x=list(range(t+1)), y=mass_log[:t+1]), # Mass Plot
                go.Heatmap(z=naive_hist[t]), # Result Copy
                go.Heatmap(z=mamba_hist[t])  # Result Copy
            ],
            name=f"fr{t}",
            layout=go.Layout(
                title_text=f"Time Step {t}: [{'NOISE' if is_noise else 'SIGNAL'}] Row {row_idx} | Mass Assigned: {mass:.2f}"
            )
        ))

    fig.frames = frames
    
    # Play Button
    fig.update_layout(
        title=f"<b>Mamba Physics: Gravity Selection vs Entropy</b><br>Target Digit: {digit_label} | Blue=Signal, Red=Noise",
        height=800,
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                 "label": "Play", "method": "animate"}
            ],
            "type": "buttons",
            "showactive": False,
            "x": 0.1, "y": -0.1
        }],
        template="plotly_dark"
    )
    
    # Fix Axes
    fig.update_yaxes(autorange='reversed', row=1, col=1)
    fig.update_yaxes(autorange='reversed', row=1, col=2)
    fig.update_yaxes(autorange='reversed', row=1, col=3)
    fig.update_yaxes(autorange='reversed', row=2, col=2)
    fig.update_yaxes(autorange='reversed', row=2, col=3)
    fig.update_yaxes(range=[-0.1, 1.1], title="Mass (Importance)", row=2, col=1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mamba_gravity_simulation.html")
    fig.write_html(out_path)
    print(f"Simulation saved to {out_path}")

def main():
    print("Loading Truth (Data)...")
    train_x, train_y, test_x, test_y = get_data()
    
    # Pick a target digit
    # Index 0 is ambiguous (predicts 2). Let's try Index 1.
    target_idx = np.where(test_y == 3)[0][1] 
    target_img = test_x[target_idx].reshape(28, 28)
    
    print("Injecting Time-Entropy (Noise Stream)...")
    # Stream is a list of dicts: {'row_idx', 'data', 'type'}
    stream = create_noisy_stream(target_img, noise_level=0.8)
    
    print(f"Stream Length: {len(stream)} packets (28 Real + Noise)")
    
    # Initialize Physics Models
    mamba = GravityMamba()
    naive = NaiveAccumulator()
    
    # Run Simulation
    print("Running Physics Simulation...")
    for packet in stream:
        frame = packet['data']
        r_idx = packet['row_idx']
        
        mamba.step(frame, r_idx)
        naive.step(frame, r_idx)
        
    # Analyze Final States with Gravity Engine
    # Flatten Back to 784D
    final_mamba = mamba.h.reshape(784)
    final_naive = naive.h.reshape(784)
    
    # Load Engine for Validation
    engine = GravityEngine()
    print("Training Gravity Engine (10k samples)...")
    engine.fit(train_x[:10000], train_y[:10000]) 
    
    # Check Baseline (The Clean, Perfect '3')
    pred_clean, _ = engine.predict(target_img.reshape(1, -1))
    
    pred_mamba, _ = engine.predict(final_mamba.reshape(1, -1))
    pred_naive, _ = engine.predict(final_naive.reshape(1, -1))
    
    print("="*40)
    print(f"RESULTS for Digit 3:")
    print(f"Baseline (Clean Image): {pred_clean[0]}")
    print(f"Mamba Reconstruction:   {pred_mamba[0]}")
    print(f"Naive Reconstruction:   {pred_naive[0]}")
    print("="*40)
    
    visualize_mamba_physics(stream, mamba.history, naive.history, mamba.mass_history, 3)

if __name__ == "__main__":
    main()

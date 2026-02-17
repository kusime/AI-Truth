import os
import sys
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path to import GravityEngine
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.sec_36_gravity_engine import GravityEngine, get_data

# --- 实验核心配置 ---
OUTPUT_DIR = "output/sec_38"
NOISE_INTENSITY = 1   # 噪声强度 (0-1)，越接近1越吵
NOISE_RATIO = 0.99      # 噪声比例 (0.99 = 99% 噪声, 0.999 = 99.9% 噪声)
TARGET_DIGIT = 3        # 我们要从混沌中吸取的“真理”

# ==========================================
# PART 1: The Physics of "Selection"
# ==========================================

class MassSelector:
    """
    物理引擎的“守门人”：计算流入数据的“引力质量”。
    """
    @staticmethod
    def calculate_mass(frame):
        max_val = np.max(frame)
        
        # 规则 A：虚无没有质量
        if max_val < 0.2:
            return 0.0
            
        # 规则 B：计算全变分 (Total Variation)
        # 信号是平滑的（TV低），噪声是剧烈抖动的（TV高）
        diffs = np.abs(np.diff(frame))
        tv = np.sum(diffs)
        
        # 极限挑战建议：
        # 如果噪声达到 99.9%，这里建议调得更严苛（比如 3.5）
        if tv > 4.2: 
            return 0.0 # 判定为熵（垃圾），离心甩飞
            
        # 规则 C：判定为有结构的真理
        return 1.0

class GravityMamba:
    """
    选择性引力累加器。
    模拟公式：h_t = h_{t-1} + Mass_t * (x_t - h_{t-1})
    """
    def __init__(self, shape=(28, 28)):
        self.shape = shape
        self.h = np.zeros(shape)
        self.history = []
        self.mass_history = []
        
    def step(self, frame, row_idx):
        # 1. 计算引力质量
        mass = MassSelector.calculate_mass(frame)
        self.mass_history.append(mass)
        
        # 2. 状态更新（引力坍缩）
        current_row_val = self.h[row_idx, :]
        new_row_val = current_row_val * (1 - mass) + frame * mass
        self.h[row_idx, :] = new_row_val
        
        self.history.append(self.h.copy())

class NaiveAccumulator:
    """
    传统朴素模型：全盘接收，没有物理过滤。
    """
    def __init__(self, shape=(28, 28)):
        self.shape = shape
        self.h = np.zeros(shape)
        self.history = []
        
    def step(self, frame, row_idx):
        self.h[row_idx, :] = frame 
        self.history.append(self.h.copy())


# ==========================================
# PART 2: The Stream Generation (极限重构版)
# ==========================================

def create_noisy_stream(digit_img, noise_ratio=0.8, noise_level=0.5):
    """
    生成一个包含极端噪声的数据流。
    noise_ratio: 噪声占总数据包的比例。
    """
    stream = []
    # 计算每个信号帧之间需要插入多少个噪声帧
    # 比例公式：N_noise / (1 + N_noise) = ratio -> N_noise = ratio / (1 - ratio)
    num_noise_per_signal = int(noise_ratio / (1 - noise_ratio)) if noise_ratio < 1 else 1000

    for r in range(28):
        # 1. 注入真实信号（真理）
        stream.append({
            'type': 'signal',
            'row_idx': r,
            'data': digit_img[r, :]
        })
        
        # 2. 注入密集噪声（熵增）
        for _ in range(num_noise_per_signal):
            noise_row = np.random.rand(28) * noise_level 
            target_r = np.random.randint(0, 28) # 随机干扰任意行
            
            stream.append({
                'type': 'noise',
                'row_idx': target_r,
                'data': noise_row
            })
            
    return stream

# ==========================================
# PART 3: Visualization (保持不变)
# ==========================================

def visualize_mamba_physics(stream, mamba_hist, naive_hist, mass_log, digit_label):
    print(f"正在生成动画 (总步数: {len(stream)})...")
    steps = len(stream)
    
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
    
    # Heatmap setup with explicit range and colorscale
    # We use 'Greys_r' so 0 is Black (background) and 1 is White (signal)
    hm_config = dict(colorscale='Greys_r', zmin=0, zmax=1, showscale=False)

    fig.add_trace(go.Heatmap(z=np.zeros((1, 28)), **hm_config), row=1, col=1)
    fig.add_trace(go.Heatmap(z=np.zeros((28, 28)), **hm_config), row=1, col=2)
    fig.add_trace(go.Heatmap(z=np.zeros((28, 28)), **hm_config), row=1, col=3)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', line=dict(color='cyan', width=2)), row=2, col=1)
    fig.add_trace(go.Heatmap(z=np.zeros((28, 28)), **hm_config), row=2, col=2)
    fig.add_trace(go.Heatmap(z=np.zeros((28, 28)), **hm_config), row=2, col=3)

    frames = []
    # Sampling for performance
    sample_rate = max(1, steps // 200) 
    
    for t in range(0, steps, sample_rate):
        packet = stream[t]
        frames.append(go.Frame(
            data=[
                go.Heatmap(z=packet['data'].reshape(1, 28), **hm_config),
                go.Heatmap(z=naive_hist[t], **hm_config),
                go.Heatmap(z=mamba_hist[t], **hm_config),
                go.Scatter(x=list(range(t+1)), y=mass_log[:t+1]),
                go.Heatmap(z=naive_hist[t], **hm_config),
                go.Heatmap(z=mamba_hist[t], **hm_config)
            ],
            name=f"fr{t}",
            layout=go.Layout(
                title_text=f"Step {t}/{steps} | Type: {packet['type'].upper()} | Mass: {mass_log[t]:.2f}"
            )
        ))

    fig.frames = frames
    
    # Update layout and Axes properly
    fig.update_layout(
        title=f"<b>Gravity Mamba vs 99% Noise</b> | Target: {digit_label} | Ratio: {NOISE_RATIO*100}%",
        height=900,
        updatemenus=[{"buttons": [{"args": [None, {"frame": {"duration": 50, "redraw": True}}], "label": "Play", "method": "animate"}], "type": "buttons"}],
        template="plotly_dark",
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    # Explicitly set Axis Ranges for Heatmaps (Top-Left Origin convention)
    # Row 1, Col 1 (1x28): range [0.5, -0.5] means top is -0.5, bottom is 0.5. y=0 is center.
    fig.update_yaxes(range=[0.5, -0.5], row=1, col=1)
    
    # Other Heatmaps (28x28): range [27.5, -0.5] puts y=0 at top, y=27 at bottom.
    for r in [1, 2]:
        for c in [1, 2, 3]:
            if (r == 1 and c == 1) or (r == 2 and c == 1):
                continue
            fig.update_yaxes(range=[27.5, -0.5], row=r, col=c)
    
    # Scatter Axis (Mass Log)
    fig.update_yaxes(range=[-0.1, 1.1], title="Mass (Gravity)", row=2, col=1)
    fig.update_xaxes(title="Time Step", row=2, col=1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mamba_gravity_simulation.html")
    fig.write_html(out_path)
    print(f"实验报告已生成: {out_path}")

def main():
    train_x, train_y, test_x, test_y = get_data()
    target_idx = np.where(test_y == TARGET_DIGIT)[0][1] 
    target_img = test_x[target_idx].reshape(28, 28)
    
    # 执行极限噪声注入
    stream = create_noisy_stream(target_img, noise_ratio=NOISE_RATIO, noise_level=NOISE_INTENSITY)
    
    mamba = GravityMamba()
    naive = NaiveAccumulator()
    
    print(f"开始物理模拟，当前噪声占比: {NOISE_RATIO*100}%...")
    for packet in stream:
        mamba.step(packet['data'], packet['row_idx'])
        naive.step(packet['data'], packet['row_idx'])
        
    print("验证推理结果...")
    engine = GravityEngine()
    engine.fit(train_x[:10000], train_y[:10000]) 
    
    pred_mamba, _ = engine.predict(mamba.h.reshape(1, -1))
    pred_naive, _ = engine.predict(naive.h.reshape(1, -1))
    
    print("="*40)
    print(f"极限测试结果 (噪声 {NOISE_RATIO*100}%):")
    print(f"Mamba (引力过滤) 预测结果: {pred_mamba[0]}  {'✅' if pred_mamba[0]==TARGET_DIGIT else '❌'}")
    print(f"Naive (朴素集成) 预测结果: {pred_naive[0]}  {'✅' if pred_naive[0]==TARGET_DIGIT else '❌'}")
    print("="*40)
    
    visualize_mamba_physics(stream, mamba.history, naive.history, mamba.mass_history, TARGET_DIGIT)

if __name__ == "__main__":
    main()
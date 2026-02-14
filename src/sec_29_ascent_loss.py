"""
Section 29: 升维的代价 - 维度诅咒与噪声淹没
The Cost of Ascension - Curse of Dimensionality & Noise Drowning

(Section 28的镜像)
Section 28结论: 在理想真空(补0)中, 升维是无损的 (100%保持)
Section 29问题: 在真实环境(含噪声)中, 升维会导致信息被"稀释"或"淹没"吗?

假设:
宇宙充满了背景噪声(量子涨落/热噪声).
当我们把3D物体放入1000D空间时, 它同时也暴露在了1000个维度的噪声中.
信号(Signal)集中在3个维度.
噪声(Noise)分布在1000个维度.
SNR (信噪比) 会随维度增加而剧烈下降.

实验设计:
1. 生成3D瑞士卷 (信号).
2. 嵌入到 N 维空间 (N = 3 -> 1000).
3. 添加各向同性高斯噪声 (模拟真实环境).
4. 尝试从强噪声的高维数据中"提取"原始结构 (Isomap降维回2D).
5. 计算提取出的结构与真理的偏差 (信息损失).

预期曲线:
维度越高 -> 噪声总能量越大 -> 有效信息越难提取 -> 损失率上升.
"""

import os
import time

import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances

os.makedirs('output/sec_29', exist_ok=True)

def generate_signal_in_noise(n_samples=500, total_dim=3, noise_level=0.5):
    """
    生成高维含噪数据
    Signal: 3D Swiss Roll (fixed energy)
    Noise: N-dim Gaussian (energy scales with dim)
    """
    # 1. 信号 (3D)
    X_signal_3d, t = make_swiss_roll(n_samples=n_samples, noise=0.0, random_state=42)
    # 归一化信号能量,使其具有也就是单位方差附近,方便控制SNR
    X_signal_3d = (X_signal_3d - X_signal_3d.mean(axis=0)) / X_signal_3d.std()
    
    # 2. 嵌入 (Pad with zeros)
    X_full = np.zeros((n_samples, total_dim))
    X_full[:, :3] = X_signal_3d
    
    # 3. 噪声 (Filling ALL dimensions)
    # sigma = noise_level
    np.random.seed(42)
    Noise = np.random.normal(0, noise_level, (n_samples, total_dim))
    
    # 合成
    X_noisy = X_full + Noise
    
    return X_noisy, t, X_signal_3d

def measure_information_retrieval(X_noisy, t_true, n_neighbors=15):
    """
    尝试从噪声中恢复流形结构
    计算恢复后的结构与真实拓扑(t)的相关性
    """
    try:
        # 使用Isomap尝试捕捉流形 (降噪能力测试)
        # 我们知道它本质是2D流形,所以目标设为2D
        iso = Isomap(n_neighbors=n_neighbors, n_components=2)
        X_recovered = iso.fit_transform(X_noisy)
        
        # 评测: 重构出的坐标距离 与 真实的参数空间距离(t) 的相关性
        # 或者更简单: 比较 X_recovered 的 pairwise distance 与 
        # 纯净信号的 pairwise geodesic distance (在此用t近似或纯净Isomap)
        
        # 这里我们用一个简化的指标: 
        # 恢复后的2D结构 internal distance vs 纯净3D信号的距离 (Section 28 verified 3D->2D is ~0.98 in clean)
        # 如果噪声太大, 这个相关性会掉得很厉害
        
        dist_recovered = pairwise_distances(X_recovered)
        
        # 比较对象: 纯净信号的距离矩阵 (or t)
        # 这里用 t (由内而外) 比较准确, 因为 t 代表了"绝对真理"的拓扑位置
        dist_true = pairwise_distances(t_true.reshape(-1, 1))
        
        # 归一化
        d1 = (dist_recovered - dist_recovered.mean()) / dist_recovered.std()
        d2 = (dist_true - dist_true.mean()) / dist_true.std()
        
        corr = np.corrcoef(d1.flatten(), d2.flatten())[0, 1]
        
        # 基准线: 在纯净3D下的 Isomap(3D->2D) 对 t 的相关性大约是 0.99
        # 我们把这个作为 100% (Base)
        # Information Retention = corr / base_corr
        
        return corr
        
    except Exception as e:
        # 崩了 (可能是噪声太大导致断裂)
        return 0

def run_experiment():
    print(f"{'='*60}")
    print("Section 29: 升维的代价 (噪声淹没测试)")
    print(f"{'='*60}")
    
    dims = [3, 5, 10, 20, 50, 100, 200, 500, 1000]
    noisy_level = 0.3 # 噪声强度 (相对于信号标准差1.0)
    
    # 计算基准 (3D Clean) 用于归一化
    # 注意: Isomap本身在3D就有微小损耗(Section 28验证过是0.98左右)
    # 这一步是为了把起点的3D Scale到 100% (Loss=0%)
    X_clean, t, _ = generate_signal_in_noise(500, 3, 0.0)
    base_corr = measure_information_retrieval(X_clean, t)
    print(f"基准保持率 (3D Clean Base): {base_corr:.4f}")
    
    results = []
    
    # 1. 测试噪声环境 (模拟现实)
    print("\n[测试1] 噪声环境 (Noise=0.3)...")
    noisy_losses = []
    snrs = []
    for d in dims:
        X_noisy, t, _ = generate_signal_in_noise(500, d, noisy_level)
        corr = measure_information_retrieval(X_noisy, t)
        retention = max(0, corr / base_corr)
        loss = 1.0 - retention
        
        # SNR Calculation
        snr = 3.0 / (d * (noisy_level**2))
        
        noisy_losses.append(loss)
        snrs.append(snr)
        print(f"  维度 {d:>4d} | SNR: {snr:.4f} | 损失: {loss*100:.2f}%")
        
    # 2. 测试纯净环境 (对照组)
    print("\n[测试2] 纯净环境 (Noise=0.0)...")
    clean_losses = []
    for d in dims:
        # Noise=0.0 (纯净升维)
        X_pure, t, _ = generate_signal_in_noise(500, d, 0.0)
        corr = measure_information_retrieval(X_pure, t)
        retention = max(0, corr / base_corr)
        loss = 1.0 - retention
        
        clean_losses.append(loss)
        print(f"  维度 {d:>4d} | 纯净 | 损失: {loss*100:.2f}%")
        
    return dims, noisy_losses, clean_losses, snrs, noisy_level

def create_viz(dims, noisy_losses, clean_losses, snrs, noise_level):
    fig = go.Figure()
    
    # 曲线1: 噪声环境损失 (红线 - 现实)
    fig.add_trace(go.Scatter(
        x=dims, y=noisy_losses,
        mode='lines+markers',
        name='现实升维损失 (含噪声)',
        line=dict(color='#ff0055', width=4),
        marker=dict(size=10, symbol='circle')
    ))
    
    # 曲线2: 纯净环境损失 (绿线 - 理想)
    fig.add_trace(go.Scatter(
        x=dims, y=clean_losses,
        mode='lines+markers',
        name='理想升维损失 (无噪声)',
        line=dict(color='#00ff88', width=4),
        marker=dict(size=10, symbol='diamond')
    ))
    
    # 曲线3: SNR (辅助线)
    max_snr = snrs[0]
    snr_norm = [s/max_snr for s in snrs]
    
    fig.add_trace(go.Scatter(
        x=dims, y=snr_norm,
        mode='lines',
        name='信噪比衰减 (SNR Decay)',
        line=dict(color='#00bdff', width=2, dash='dot')
    ))

    fig.update_layout(
        title=f'Section 29: 升维的代价 - 噪声淹没效应 vs 理想对照',
        template='plotly_dark',
        height=600,
        xaxis=dict(
            title='所在维度 (Dimension)',
            type='log',
            tickvals=[3, 10, 50, 100, 500, 1000],
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title='信息损失率 (Loss Ratio)',
            range=[-0.05, 1.05],
            gridcolor='rgba(255,255,255,0.1)'
        ),
        legend=dict(x=0.05, y=0.95),
        annotations=[
            dict(x=np.log10(1000), y=noisy_losses[-1], text=f"噪声淹没: {noisy_losses[-1]*100:.0f}%", showarrow=True, arrowhead=1, ax=-60, ay=10, font=dict(color='#ff0055')),
            dict(x=np.log10(1000), y=clean_losses[-1], text=f"纯净保持: 损失Only {clean_losses[-1]*100:.1f}%", showarrow=True, arrowhead=1, ax=-60, ay=-20, font=dict(color='#00ff88'))
        ]
    )
    
    fig.write_html('output/sec_29/ascent_loss.html')
    print("\n可视化已生成: output/sec_29/ascent_loss.html")

if __name__ == '__main__':
    dims, n_loss, c_loss, snrs, nl = run_experiment()
    create_viz(dims, n_loss, c_loss, snrs, nl)

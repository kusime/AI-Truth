"""
Section 15: n+1维不变性 - 单位标尺与几何不变量
n+1 Dimensional Invariance - Unit Scale and Geometric Invariants

核心洞察:
秩-1矩阵是n维空间的"单位标尺"(最小线性不可分单元)
在n+1维输出空间中,单样本和多样本都是"点",只是位置不同
"单样本 vs 多样本"是n维的概念,在n+1维中它们是同类对象

验证:
1. 秩-1矩阵的最小性(不可再分)
2. n+1维空间的几何不变性
3. 单样本=多样本(在n+1维视角)
"""

import os

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

# 创建输出目录
os.makedirs('output/sec_15', exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

# ============================================
# Part 1: 验证秩-1矩阵是"单位标尺"
# ============================================

def verify_rank_one_unit_scale():
    """验证秩-1矩阵的最小性"""
    set_seed(42)
    
    dim = 8  # 使用小维度便于可视化
    
    # 创建秩-1矩阵
    v = torch.randn(dim)
    k = torch.randn(dim)
    
    # 外积
    rank_one_matrix = torch.outer(v, k)
    
    # 验证秩
    rank = torch.linalg.matrix_rank(rank_one_matrix).item()
    
    # SVD分解
    U, S, Vh = torch.linalg.svd(rank_one_matrix)
    
    print(f"\n{'='*80}")
    print("Part 1: 秩-1矩阵是n维空间的'单位标尺'")
    print(f"{'='*80}")
    print(f"矩阵维度: {dim}×{dim}")
    print(f"矩阵秩: {rank}")
    print(f"奇异值: {S.numpy()}")
    print(f"非零奇异值数量: {(S > 1e-6).sum().item()}")
    print(f"\n✓ 秩-1矩阵只有1个非零奇异值,是最小的线性不可分单元")
    
    return {
        'matrix': rank_one_matrix.numpy(),
        'rank': rank,
        'singular_values': S.numpy(),
        'v': v.numpy(),
        'k': k.numpy()
    }

# ============================================
# Part 2: n+1维空间的几何不变性
# ============================================

def verify_n_plus_1_invariance():
    """验证在n+1维输出空间中,单样本和多样本都是向量"""
    set_seed(42)
    
    dim = 64
    n_samples = 10
    lr = 1.0 / dim
    
    # 数据
    query = torch.randn(dim)
    keys = torch.randn(n_samples, dim)
    values = torch.randn(n_samples, dim)
    
    # 单样本在n+1维的表示
    single_samples = []
    for i in range(n_samples):
        # n维: 秩-1矩阵
        delta_W = lr * torch.outer(values[i], keys[i])
        
        # n+1维: 向量
        delta_y = delta_W @ query
        
        # 验证等价性
        delta_y_direct = lr * (query @ keys[i]) * values[i]
        
        diff = torch.norm(delta_y - delta_y_direct).item()
        
        single_samples.append({
            'delta_y': delta_y.numpy(),
            'norm': torch.norm(delta_y).item(),
            'diff': diff
        })
    
    # 多样本在n+1维的表示
    delta_y_total = sum([torch.tensor(s['delta_y']) for s in single_samples])
    
    print(f"\n{'='*80}")
    print("Part 2: n+1维空间的几何不变性")
    print(f"{'='*80}")
    print(f"输出空间维度: {dim}")
    print(f"\n单样本 (n+1维向量):")
    for i, s in enumerate(single_samples[:3]):
        print(f"  样本 {i+1}: ||Δy|| = {s['norm']:.4f}, 验证误差 = {s['diff']:.2e}")
    print(f"  ...")
    
    print(f"\n多样本 (n+1维向量的和):")
    print(f"  ||Σ Δy|| = {torch.norm(delta_y_total).item():.4f}")
    
    print(f"\n✓ 单样本和多样本都是{dim}维向量,在n+1维空间中是同类对象")
    
    return {
        'single_samples': single_samples,
        'multi_sample_norm': torch.norm(delta_y_total).item(),
        'dim': dim
    }

# ============================================
# Part 3: 维度相对性验证
# ============================================

def verify_dimensional_relativity():
    """验证'单样本vs多样本'是n维概念,在n+1维中等价"""
    set_seed(42)
    
    dim = 64
    n_samples_list = [1, 2, 5, 10, 20, 50]
    lr = 1.0 / dim
    
    # 数据
    query = torch.randn(dim)
    keys = torch.randn(max(n_samples_list), dim)
    values = torch.randn(max(n_samples_list), dim)
    
    results = []
    
    for n in n_samples_list:
        # n维视角: 多个秩-1矩阵
        delta_W_total = torch.zeros(dim, dim)
        for i in range(n):
            delta_W_total += lr * torch.outer(values[i], keys[i])
        
        rank_n_dim = torch.linalg.matrix_rank(delta_W_total).item()
        
        # n+1维视角: 向量
        delta_y_total = delta_W_total @ query
        norm_n_plus_1_dim = torch.norm(delta_y_total).item()
        
        results.append({
            'n_samples': n,
            'rank_n_dim': rank_n_dim,
            'norm_n_plus_1_dim': norm_n_plus_1_dim
        })
    
    print(f"\n{'='*80}")
    print("Part 3: 维度相对性 - 单样本vs多样本")
    print(f"{'='*80}")
    print(f"{'样本数':>8} | {'n维空间(秩)':>15} | {'n+1维空间(||Δy||)':>20}")
    print(f"{'-'*8}-+-{'-'*15}-+-{'-'*20}")
    for r in results:
        print(f"{r['n_samples']:>8} | {r['rank_n_dim']:>15} | {r['norm_n_plus_1_dim']:>20.4f}")
    
    print(f"\n观察:")
    print(f"  - n维空间: 秩随样本数增加 (1→{results[-1]['rank_n_dim']})")
    print(f"  - n+1维空间: 都是{dim}维向量,只是大小不同")
    print(f"\n✓ '单样本vs多样本'是n维的区别,在n+1维中都是'点'")
    
    return results

# ============================================
# Part 4: 可视化
# ============================================

def create_visualizations():
    """创建所有可视化"""
    
    # 运行验证
    rank_one_data = verify_rank_one_unit_scale()
    invariance_data = verify_n_plus_1_invariance()
    relativity_data = verify_dimensional_relativity()
    
    # ============================================
    # 可视化 1: 秩-1矩阵热图
    # ============================================
    
    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '秩-1矩阵 (v ⊗ k^T)',
            '奇异值分解 - 只有1个非零奇异值'
        ),
        specs=[[{'type': 'heatmap'}, {'type': 'bar'}]]
    )
    
    # 左图: 矩阵热图
    fig1.add_trace(
        go.Heatmap(
            z=rank_one_data['matrix'],
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            hovertemplate='行 %{y}<br>列 %{x}<br>值 %{z:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 右图: 奇异值
    fig1.add_trace(
        go.Bar(
            x=list(range(1, len(rank_one_data['singular_values']) + 1)),
            y=rank_one_data['singular_values'],
            marker=dict(
                color=['#ff0055'] + ['#333333'] * (len(rank_one_data['singular_values']) - 1)
            ),
            hovertemplate='奇异值 %{x}<br>大小 %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig1.update_xaxes(title_text='列', row=1, col=1)
    fig1.update_yaxes(title_text='行', row=1, col=1)
    fig1.update_xaxes(title_text='奇异值索引', row=1, col=2)
    fig1.update_yaxes(title_text='大小', type='log', row=1, col=2)
    
    fig1.update_layout(
        title={
            'text': '秩-1矩阵: n维空间的"单位标尺"<br><sub>最小的线性不可分单元,只有1个非零奇异值</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#00f2ff'}
        },
        template='plotly_dark',
        height=600,
        showlegend=False,
        font=dict(family='Fira Code, monospace')
    )
    
    fig1.write_html('output/sec_15/rank_one_unit_scale.html')
    print(f"\n✅ 可视化 1: output/sec_15/rank_one_unit_scale.html")
    
    # ============================================
    # 可视化 2: n+1维空间的向量分布
    # ============================================
    
    fig2 = go.Figure()
    
    # 单样本向量的模
    norms = [s['norm'] for s in invariance_data['single_samples']]
    
    fig2.add_trace(go.Bar(
        x=list(range(1, len(norms) + 1)),
        y=norms,
        marker=dict(
            color=norms,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='||Δy||')
        ),
        hovertemplate='样本 %{x}<br>||Δy|| = %{y:.4f}<extra></extra>'
    ))
    
    # 添加平均线
    avg_norm = np.mean(norms)
    fig2.add_hline(
        y=avg_norm,
        line_dash="dash",
        line_color="#ff0055",
        annotation_text=f"平均值 = {avg_norm:.4f}",
        annotation_position="right"
    )
    
    fig2.update_layout(
        title={
            'text': f'单样本在n+1维空间的表示<br><sub>每个样本都是{invariance_data["dim"]}维向量,只是大小不同</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#00f2ff'}
        },
        xaxis_title='样本索引',
        yaxis_title='向量模 ||Δy||',
        template='plotly_dark',
        height=600,
        font=dict(family='Fira Code, monospace')
    )
    
    fig2.write_html('output/sec_15/n_plus_1_vectors.html')
    print(f"✅ 可视化 2: output/sec_15/n_plus_1_vectors.html")
    
    # ============================================
    # 可视化 3: 维度相对性
    # ============================================
    
    fig3 = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'n维空间: 秩随样本数增加',
            'n+1维空间: 都是向量,只是大小不同'
        )
    )
    
    n_samples = [r['n_samples'] for r in relativity_data]
    ranks = [r['rank_n_dim'] for r in relativity_data]
    norms = [r['norm_n_plus_1_dim'] for r in relativity_data]
    
    # 左图: n维空间的秩
    fig3.add_trace(
        go.Scatter(
            x=n_samples,
            y=ranks,
            mode='lines+markers',
            name='矩阵秩',
            line=dict(color='#ff0055', width=3),
            marker=dict(size=10),
            hovertemplate='样本数 %{x}<br>秩 %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 右图: n+1维空间的向量模
    fig3.add_trace(
        go.Scatter(
            x=n_samples,
            y=norms,
            mode='lines+markers',
            name='向量模',
            line=dict(color='#00f2ff', width=3),
            marker=dict(size=10),
            hovertemplate='样本数 %{x}<br>||Δy|| %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig3.update_xaxes(title_text='样本数量', row=1, col=1)
    fig3.update_xaxes(title_text='样本数量', row=1, col=2)
    fig3.update_yaxes(title_text='矩阵秩', row=1, col=1)
    fig3.update_yaxes(title_text='||Δy||', row=1, col=2)
    
    fig3.update_layout(
        title={
            'text': '维度相对性: 单样本vs多样本<br><sub>n维: 不同秩的矩阵 | n+1维: 不同大小的向量</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#00f2ff'}
        },
        template='plotly_dark',
        height=600,
        showlegend=True,
        font=dict(family='Fira Code, monospace')
    )
    
    fig3.write_html('output/sec_15/dimensional_relativity.html')
    print(f"✅ 可视化 3: output/sec_15/dimensional_relativity.html")
    
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    print(f"✅ 秩-1矩阵是n维空间的'单位标尺'(最小线性不可分单元)")
    print(f"✅ 在n+1维输出空间中,单样本和多样本都是向量")
    print(f"✅ '单样本vs多样本'是n维的概念,在n+1维中是同类对象")
    print(f"\n核心洞察:")
    print(f"  n维空间: 单样本=秩-1矩阵, 多样本=多个秩-1矩阵的和")
    print(f"  n+1维空间: 单样本=向量, 多样本=向量的和=另一个向量")
    print(f"  → 在n+1维中,它们都是'点',只是位置不同 ✓")

if __name__ == '__main__':
    create_visualizations()

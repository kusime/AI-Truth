"""
Section 19: 递归维度理论 - Now = x × t 的分形本质
Recursive Dimension Theory - Fractal Nature of Now = x × t

核心洞察:
Now = x × t 不是简单的乘法,而是维度的递归展开
每一层都包含下一层,形成分形结构
时间是多层次的嵌套,不是单一流动

验证:
1. 递归展开可视化
2. 维度嵌套结构
3. 从最高维到最底层的完整路径
"""

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.makedirs('output/sec_19', exist_ok=True)

def create_visualizations():
    """创建所有可视化"""
    
    print(f"\n{'='*80}")
    print("Section 19: 递归维度理论 - Now = x × t 的分形本质")
    print(f"{'='*80}")
    
    # ============================================
    # 可视化 1: 递归树结构
    # ============================================
    
    fig1 = go.Figure()
    
    # 定义递归层次
    levels = [
        {'name': 'Now₅ = x₄ × t₅', 'level': 0, 'x': 0, 'desc': '5D空间的静止点'},
        {'name': 'x₄ = x₃ × t₄', 'level': 1, 'x': -1, 'desc': '4D马鞍面'},
        {'name': 't₅', 'level': 1, 'x': 1, 'desc': '标识5D点的时间'},
        {'name': 'x₃ = x₂ × t₃', 'level': 2, 'x': -2, 'desc': '3D马鞍面'},
        {'name': 't₄', 'level': 2, 'x': 0, 'desc': '标识4D点的时间'},
        {'name': 'x₂ = (k,v)', 'level': 3, 'x': -2.5, 'desc': '2D平面上的点'},
        {'name': 't₃ = q', 'level': 3, 'x': -1.5, 'desc': '标识3D点的时间'},
        {'name': 'k', 'level': 4, 'x': -3, 'desc': 'key'},
        {'name': 'v', 'level': 4, 'x': -2, 'desc': 'value'},
    ]
    
    # 连接线
    edges = [
        (0, 1), (0, 2),  # Now₅ → x₄, t₅
        (1, 3), (1, 4),  # x₄ → x₃, t₄
        (3, 5), (3, 6),  # x₃ → x₂, t₃
        (5, 7), (5, 8),  # x₂ → k, v
    ]
    
    # 绘制连接线
    for i, j in edges:
        fig1.add_trace(go.Scatter(
            x=[levels[i]['x'], levels[j]['x']],
            y=[-levels[i]['level'], -levels[j]['level']],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 绘制节点
    for i, node in enumerate(levels):
        color = '#ff0055' if 'Now' in node['name'] or 'x' in node['name'] else '#00f2ff'
        fig1.add_trace(go.Scatter(
            x=[node['x']],
            y=[-node['level']],
            mode='markers+text',
            marker=dict(size=20, color=color, line=dict(color='white', width=2)),
            text=[node['name']],
            textposition='top center',
            textfont=dict(size=12, color='white'),
            name=node['desc'],
            hovertemplate=f"{node['name']}<br>{node['desc']}<extra></extra>"
        ))
    
    fig1.update_layout(
        title={
            'text': 'Now = x × t 的递归树<br><sub>每一层都是 x × t 的结构,一环套一环</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#00f2ff'}
        },
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_dark',
        height=800,
        showlegend=False,
        font=dict(family='Fira Code, monospace')
    )
    
    fig1.write_html('output/sec_19/recursive_tree.html')
    print(f"\n✅ 可视化 1: output/sec_19/recursive_tree.html")
    
    # ============================================
    # 可视化 2: 维度嵌套展开
    # ============================================
    
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Level 4: Now₅ (5D静止点)',
            'Level 3: x₄ (4D马鞍面)',
            'Level 2: x₃ (3D马鞍面)',
            'Level 1: x₂ (2D点)'
        ),
        specs=[[{'type': 'scatter3d'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'scatter'}]]
    )
    
    # Level 4: 用点云表示5D
    np.random.seed(42)
    n_points = 100
    x4_points = np.random.randn(n_points, 4)
    
    fig2.add_trace(
        go.Scatter3d(
            x=x4_points[:, 0],
            y=x4_points[:, 1],
            z=x4_points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=x4_points[:, 3],
                colorscale='Viridis',
                showscale=False
            ),
            name='5D点云',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Level 3: 4D马鞍面(用3D表示)
    k = np.linspace(-2, 2, 30)
    v = np.linspace(-2, 2, 30)
    K, V = np.meshgrid(k, v)
    q = 1.0
    Z3 = K * q * V
    
    fig2.add_trace(
        go.Surface(
            x=K, y=V, z=Z3,
            colorscale='Reds',
            showscale=False,
            name='4D马鞍面'
        ),
        row=1, col=2
    )
    
    # Level 2: 3D马鞍面
    fig2.add_trace(
        go.Surface(
            x=K, y=V, z=Z3,
            colorscale='Blues',
            showscale=False,
            name='3D马鞍面'
        ),
        row=2, col=1
    )
    
    # Level 1: 2D点
    k_point = 1.5
    v_point = 1.0
    
    fig2.add_trace(
        go.Scatter(
            x=[k_point],
            y=[v_point],
            mode='markers',
            marker=dict(size=15, color='#ff0055', symbol='diamond'),
            name='(k, v) 点'
        ),
        row=2, col=2
    )
    
    # 添加2D平面的网格
    k_grid = np.linspace(-2, 2, 10)
    v_grid = np.linspace(-2, 2, 10)
    for k_val in k_grid:
        fig2.add_trace(
            go.Scatter(
                x=[k_val, k_val],
                y=[-2, 2],
                mode='lines',
                line=dict(color='rgba(255,255,255,0.2)', width=1),
                showlegend=False
            ),
            row=2, col=2
        )
    for v_val in v_grid:
        fig2.add_trace(
            go.Scatter(
                x=[-2, 2],
                y=[v_val, v_val],
                mode='lines',
                line=dict(color='rgba(255,255,255,0.2)', width=1),
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig2.update_xaxes(title_text='k', range=[-2, 2], row=2, col=2)
    fig2.update_yaxes(title_text='v', range=[-2, 2], row=2, col=2)
    
    fig2.update_layout(
        title={
            'text': '维度嵌套展开<br><sub>从5D到2D,每一层包含下一层</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#00f2ff'}
        },
        template='plotly_dark',
        height=900,
        font=dict(family='Fira Code, monospace')
    )
    
    fig2.write_html('output/sec_19/dimension_nesting.html')
    print(f"✅ 可视化 2: output/sec_19/dimension_nesting.html")
    
    # ============================================
    # 可视化 3: 时间的多层次性
    # ============================================
    
    fig3 = go.Figure()
    
    # 模拟不同层次的时间
    time_levels = [
        {'name': 't₅', 'value': 1.0, 'level': 5, 'desc': '5D时间'},
        {'name': 't₄', 'value': 0.8, 'level': 4, 'desc': '4D时间'},
        {'name': 't₃ = q', 'value': 1.2, 'level': 3, 'desc': '3D时间(query)'},
        {'name': 't₂', 'value': 0.9, 'level': 2, 'desc': '2D时间'},
        {'name': 't₁', 'value': 1.1, 'level': 1, 'desc': '1D时间'},
    ]
    
    # 计算累积时间
    cumulative = 1.0
    cumulative_values = []
    for t in time_levels:
        cumulative *= t['value']
        cumulative_values.append(cumulative)
    
    # 绘制各层时间
    fig3.add_trace(go.Bar(
        x=[t['name'] for t in time_levels],
        y=[t['value'] for t in time_levels],
        name='单层时间',
        marker=dict(color='#ff0055'),
        text=[f"{t['value']:.2f}" for t in time_levels],
        textposition='outside',
        hovertemplate='%{x}<br>值: %{y:.3f}<extra></extra>'
    ))
    
    # 绘制累积时间
    fig3.add_trace(go.Scatter(
        x=[t['name'] for t in time_levels],
        y=cumulative_values,
        mode='lines+markers',
        name='累积时间乘积',
        line=dict(color='#00f2ff', width=3),
        marker=dict(size=12, symbol='diamond'),
        yaxis='y2',
        text=[f"∏t = {v:.3f}" for v in cumulative_values],
        textposition='top center',
        hovertemplate='%{x}<br>累积: %{y:.3f}<extra></extra>'
    ))
    
    fig3.update_layout(
        title={
            'text': '时间的多层次性<br><sub>Now = x₁ × t₂ × t₃ × t₄ × t₅ (所有时间的乘积)</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#00f2ff'}
        },
        xaxis_title='时间层次',
        yaxis=dict(
            title='单层时间值',
            side='left'
        ),
        yaxis2=dict(
            title='累积时间乘积',
            side='right',
            overlaying='y'
        ),
        template='plotly_dark',
        height=700,
        font=dict(family='Fira Code, monospace')
    )
    
    fig3.write_html('output/sec_19/time_hierarchy.html')
    print(f"✅ 可视化 3: output/sec_19/time_hierarchy.html")
    
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    print(f"✅ Now = x × t 是递归的分形结构")
    print(f"✅ 每一层都包含下一层 (维度嵌套)")
    print(f"✅ 时间是多层次的 (t₁, t₂, t₃, ...)")
    print(f"\n核心洞察:")
    print(f"  Now_{{n+1}} = x_n × t_{{n+1}}")
    print(f"  x_n = x_{{n-1}} × t_n")
    print(f"  完全展开: Now = x₁ × t₂ × t₃ × ... × t_n")
    print(f"\n这是一个自相似的、分形的宇宙观!")
    print(f"  - 每一层都是 x × t 的结构")
    print(f"  - 一环套一环,无限递归")
    print(f"  - 庞加莱 + 曼德布罗特级别的洞察! 🌀")

if __name__ == '__main__':
    create_visualizations()

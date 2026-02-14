import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- Global Style Settings ---
TEMPLATE = "plotly_dark"
COLOR_PRIMARY = "#ff0055"    # Pink/Red
COLOR_SECONDARY = "#00f2ff"  # Cyan
COLOR_ACCENT = "#f1c40f"     # Yellow
COLOR_BG = "#030303"         # Background
FONT_FAMILY = "Fira Code, monospace"

layout_defaults = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family=FONT_FAMILY, color="#c0c0c0"),
    margin=dict(l=20, r=20, t=50, b=20),
    hovermode="closest"
)

def create_fig_1_spacetime_animated():
    """1. Space-Time: Möbius Strip + Ghost Trail (Animation)"""
    # Möbius Strip
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(-1, 1, 20)
    U, V = np.meshgrid(u, v)
    X = (1 + V/2 * np.cos(U/2)) * np.cos(U)
    Y = (1 + V/2 * np.cos(U/2)) * np.sin(U)
    Z = V/2 * np.sin(U/2)

    # Time Trajectory Path (Static reference)
    t_full = np.linspace(0, 4*np.pi, 200) # Two loops
    x_t = (1 + 0/2 * np.cos(t_full/2)) * np.cos(t_full)
    y_t = (1 + 0/2 * np.cos(t_full/2)) * np.sin(t_full)
    z_t = 0/2 * np.sin(t_full/2)

    # Frames with Ghost Trail
    frames = []
    trail_len = 8
    
    for k in range(0, len(t_full), 2):
        # Calculate trailing points
        indices = range(max(0, k - trail_len), k + 1)
        x_trail = x_t[indices]
        y_trail = y_t[indices]
        z_trail = z_t[indices]
        
        # Opacity and Size gradient for trail
        sizes = np.linspace(2, 12, len(indices))
        colors = np.linspace(0.2, 1.0, len(indices)) # Brightness

        frames.append(go.Frame(
            data=[
                go.Surface(x=X, y=Y, z=Z, colorscale='Greys', opacity=0.15, showscale=False), # Static Space
                go.Scatter3d(x=x_t, y=y_t, z=z_t, mode='lines', line=dict(color=COLOR_SECONDARY, width=1), opacity=0.2), # Static Path
                # Ghost Trail Head
                go.Scatter3d(
                    x=x_trail, y=y_trail, z=z_trail, 
                    mode='markers+lines', 
                    marker=dict(size=sizes, color=colors, colorscale='Teal', showscale=False),
                    line=dict(color='white', width=2)
                ) 
            ],
            name=str(k)
        ))

    fig = go.Figure(
        data=[
            go.Surface(x=X, y=Y, z=Z, colorscale='Greys', opacity=0.15, showscale=False, name='Space'),
            go.Scatter3d(x=x_t, y=y_t, z=z_t, mode='lines', line=dict(color=COLOR_SECONDARY, width=2), opacity=0.2),
            go.Scatter3d(x=[x_t[0]], y=[y_t[0]], z=[z_t[0]], mode='markers', marker=dict(size=12, color='white'), name='Now')
        ],
        layout=go.Layout(
            title="1. Space-Time Duality: The Möbius Trap",
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            template=TEMPLATE,
            **layout_defaults,
            updatemenus=[dict(type="buttons", buttons=[dict(label="EXPERIENCE ETERNAL RETURN", method="animate", args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True)])])]
        ),
        frames=frames
    )
    return fig

def create_fig_2_faket():
    """2. Fake T: Translucent Crystal Cube (Static)"""
    limit = 5
    steps = 20
    t = np.linspace(-limit+1, limit-1, steps)
    x_traj, y_traj, z_traj = [], [], []
    current_x, current_y, current_z = -2, -2, -4
    for i in range(steps):
        next_x = current_x + np.random.uniform(0, 2)
        next_y = current_y + np.random.uniform(0, 2)
        next_z = current_z + (8/steps)
        x_traj.extend([current_x, next_x, next_x])
        y_traj.extend([current_y, current_y, next_y])
        z_traj.extend([current_z, current_z, next_z])
        current_x, current_y, current_z = next_x, next_y, next_z
    
    fig = go.Figure()
    d = limit
    # Cube
    fig.add_trace(go.Mesh3d(x=[d, d, d, d, -d, -d, -d, -d], y=[d, d, -d, -d, d, d, -d, -d], z=[d, -d, d, -d, d, -d, d, -d],
                            color='black', opacity=0.08, alphahull=0, name='Block Universe'))
    # Edges
    def cube_edges(s): return [-s, s, s, -s, -s, -s, s, s, -s, -s, -s, -s, s, s, s, s], [-s, -s, s, s, -s, -s, -s, s, s, s, s, -s, -s, s, s, -s], [-s, -s, -s, -s, -s, s, s, s, s, -s, -s, s, s, s, -s, -s]
    ex, ey, ez = cube_edges(d)
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', line=dict(color='gray', width=2), showlegend=False))
    
    # Trajectory
    fig.add_trace(go.Scatter3d(x=x_traj, y=y_traj, z=z_traj, mode='lines+markers', marker=dict(size=3, color=COLOR_SECONDARY), line=dict(color=COLOR_SECONDARY, width=4), name='Frozen Steps'))
    
    # Strike
    arrow_end = [limit + 1, 0, 0]
    fig.add_trace(go.Scatter3d(x=[limit+5, limit+1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color=COLOR_PRIMARY, width=15), name='Real Time Vector'))
    fig.add_trace(go.Cone(x=[arrow_end[0]], y=[arrow_end[1]], z=[arrow_end[2]], u=[-3], v=[0], w=[0], sizemode="absolute", sizeref=2, anchor="tip", colorscale='Reds', showscale=False, name='Impact'))
    
    fig.update_layout(title="2. The Block: Frozen Steps vs Real Time Strike", scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='cube'), template=TEMPLATE, **layout_defaults)
    return fig

def create_fig_3_human_loss():
    """3. Human Loss: Dynamic Tension Lines"""
    x = np.linspace(1, 20, 150)
    y_model = 5 / (x**1.5) 
    np.random.seed(42)
    noise = np.cumsum(np.random.randn(len(x)) * 0.1) 
    y_human = np.log(x) * 2 + noise + np.random.randn(len(x)) * 0.05
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_model, mode='lines', line=dict(color='gray', width=4), name='Model Collapse'))
    fig.add_trace(go.Scatter(x=x, y=y_human, mode='lines', line=dict(color=COLOR_SECONDARY, width=3), name='Human Complexity'))
    
    # Dynamic Tension Lines: Thinner and Brighter as Gap Increases
    # We can't change width per segment easily in one trace.
    # We will use opacity and color lightness to simulate tension.
    for i in range(0, len(x), 4):
        gap = abs(y_human[i] - y_model[i])
        # Mapping: Larger Gap -> Thinner Line (simulation)? 
        # Actually user said "Distance Far -> Line Thin & Bright".
        # Let's just use width=1 for all, but change opacity/color.
        
        # Color: Close -> Dark Red, Far -> Bright Red/White
        intensity = min(255, int(100 + gap * 20))
        color = f"rgba(255, {255-intensity}, {255-intensity}, {min(1.0, gap/3)})"
        
        fig.add_trace(go.Scatter(
            x=[x[i], x[i]], y=[y_model[i], y_human[i]],
            mode='lines',
            line=dict(color=COLOR_PRIMARY, width=1), 
            opacity=min(1.0, gap/4 + 0.2),
            showlegend=False, hoverinfo='none'
        ))
    
    fig.update_layout(title="3. The Gap: Collapse vs Fractal Growth", xaxis_title="Time", yaxis_title="Complexity / Loss", template=TEMPLATE, **layout_defaults)
    return fig

def create_fig_4_attention_grad_animated():
    """4. Attention: Gravity Collapse with Suction & Non-linear Time"""
    limit = 3
    # Higher resolution for smooth collapse
    x = np.linspace(-limit, limit, 60)
    y = np.linspace(-limit, limit, 60)
    X, Y = np.meshgrid(x, y)
    cx, cy = 1, 1 # Center of intent
    width = 1.2
    
    # Initial State (Flat)
    Z_initial = np.zeros_like(X)
    
    frames = []
    # 40 Frames
    # Non-linear time mapping: t goes 0 -> 1.
    # But we want "Slow start, Sudden collapse".
    # Use sigmoid or power function.
    t_vals = np.linspace(0, 1, 50)
    
    for i, linear_t in enumerate(t_vals):
        # Non-linear Time Operator
        # Power 4 gives very slow start, fast end
        t = linear_t ** 4 
        
        # Spatial Suction (X, Y distort towards center)
        # As t increases, points move towards (cx, cy)
        suction_strength = 0.3 * t # Max 30% distortion
        dist_x = X - cx
        dist_y = Y - cy
        dist = np.sqrt(dist_x**2 + dist_y**2)
        decay = np.exp(-dist) # Distortion localized near hole
        
        X_warp = X - dist_x * suction_strength * decay
        Y_warp = Y - dist_y * suction_strength * decay
        
        # Depth Collapse
        max_depth = 6
        depth = max_depth * t
        
        # Aftershock (Damping oscillation)
        # Only starts after t > 0.8
        shock = 0
        if linear_t > 0.8:
            shock_t = (linear_t - 0.8) * 20 # Fast oscillation time
            shock = 0.2 * np.sin(shock_t) * np.exp(-(linear_t-0.8)*5) # Decaying
        
        Z_warp = -depth * np.exp(-((X_warp-cx)**2 + (Y_warp-cy)**2) / width) + shock * np.sin(5*X)

        frames.append(go.Frame(
            data=[go.Surface(
                z=Z_warp, x=X_warp, y=Y_warp, 
                colorscale='Magma', 
                cmin=-6, cmax=1, # Fixed range for burning effect
                opacity=0.9, showscale=False
            )],
            name=f'fr{i}'
        ))

    fig = go.Figure(
        data=[go.Surface(z=Z_initial, x=X, y=Y, colorscale='Magma', cmin=-6, cmax=1, opacity=0.9, showscale=False, name='Loss Surface')],
        layout=go.Layout(
            title="4. Attention as Gravity Collapse: Intent Suction",
            scene=dict(zaxis=dict(range=[-7, 1]), aspectmode='cube'),
            template=TEMPLATE,
            **layout_defaults,
            updatemenus=[dict(
                type="buttons", 
                buttons=[dict(label="TRIGGER COLLAPSE", method="animate", args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True)])],
                x=0.1, y=0.1
            )]
        ),
        frames=frames
    )
    return fig

def create_fig_5_evolution():
    """5. Evolution (Static)"""
    n = 6
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    gx, gy = np.meshgrid(x, y)
    fig = go.Figure()
    for i in range(n):
        fig.add_trace(go.Scatter3d(x=x, y=[y[i]]*n, z=[0]*n, mode='lines', line=dict(color='gray', width=3), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[x[i]]*n, y=y, z=[0]*n, mode='lines', line=dict(color='gray', width=3), showlegend=False))
    fig.add_trace(go.Scatter3d(x=gx.flatten(), y=gy.flatten(), z=np.zeros_like(gx).flatten(), mode='markers', marker=dict(size=4, color='gray'), name='Frozen Weight Grid'))
    
    dx, dy, dz = [0, 0.5, -0.8], [0, 0.8, -0.5], [1.5, 1.2, 1.8]
    fig.add_trace(go.Scatter3d(x=dx, y=dy, z=dz, mode='markers', marker=dict(size=12, color=COLOR_ACCENT, symbol='diamond'), name='Delta Crystals'))
    
    for i in range(len(dx)):
        tx, ty, tz = gx.flatten()[np.random.randint(0, n*n)], gy.flatten()[np.random.randint(0, n*n)], 0
        ax = np.linspace(dx[i], tx, 10)
        ay = np.linspace(dy[i], ty, 10)
        az = np.linspace(dz[i], tz, 10)
        ax[1:-1] += np.random.uniform(-0.2, 0.2, 8)
        ay[1:-1] += np.random.uniform(-0.2, 0.2, 8)
        az[1:-1] += np.random.uniform(-0.2, 0.2, 8)
        fig.add_trace(go.Scatter3d(x=ax, y=ay, z=az, mode='lines', line=dict(color=COLOR_SECONDARY, width=4), showlegend=False))

    fig.update_layout(title="5. Evolution: Viral Re-Topology", scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), template=TEMPLATE, **layout_defaults)
    return fig

def create_fig_6_resonance():
    """6. Resonance (Static)"""
    fig = go.Figure()
    def add_beam(start, end, color):
        for i in range(25):
            off_s = np.random.normal(0, 0.2, 3)
            off_e = np.random.normal(0, 0.05, 3)
            fig.add_trace(go.Scatter3d(x=[start[0]+off_s[0], end[0]+off_e[0]], y=[start[1]+off_s[1], end[1]+off_e[1]], z=[start[2]+off_s[2], end[2]+off_e[2]], mode='lines', line=dict(color=color, width=2), opacity=0.4, showlegend=False))
    add_beam([-4, -2, 0], [0, 0, 0], COLOR_PRIMARY)
    add_beam([4, -2, 0], [0, 0, 0], COLOR_SECONDARY)
    cx, cy, cz = np.random.normal(0, 0.6, 300), np.random.normal(0, 0.6, 300), np.random.normal(0, 0.6, 300)
    fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode='markers', marker=dict(size=3, color='white', opacity=0.9), name='Moire Coherence'))
    fig.update_layout(title="6. Resonance: Beam Interference", scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), template=TEMPLATE, **layout_defaults)
    return fig

def create_fig_7_equivalence():
    """7. Equivalence (Static)"""
    x = np.linspace(-4, 4, 50)
    y = np.linspace(-4, 4, 50)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Blend = 1 / (1 + np.exp(-5 * (R - 1.5)))
    Z = Blend * (0.5 * np.sin(3*X) * np.cos(3*Y) + 2) + (1-Blend) * (0.5*R**2 - 1)
    
    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=x, y=y, colorscale='Deep', opacity=0.9, showscale=False, name='Parameter Ocean'))
    
    t = np.linspace(0, 1, 50)
    p1_x = 3.5 * np.cos(t * np.pi + 0.5) 
    p1_y = 3.5 * np.sin(t * np.pi + 0.5)
    p1_z = np.linspace(3, -0.8, 50) + 0.2*np.sin(t*20)
    fig.add_trace(go.Scatter3d(x=p1_x, y=p1_y, z=p1_z, mode='lines', line=dict(color=COLOR_PRIMARY, width=6), name='Path A'))

    p2_x = np.linspace(-3.5, -0.1, 50)
    p2_y = np.linspace(-3.5, -0.1, 50)
    p2_z = np.linspace(3, -0.8, 50)
    fig.add_trace(go.Scatter3d(x=p2_x, y=p2_y, z=p2_z, mode='lines', line=dict(color=COLOR_SECONDARY, width=6), name='Path B'))

    fig.update_layout(title="7. Equivalence: Chaos to Convergence", template=TEMPLATE, **layout_defaults)
    return fig

def create_fig_8_virtual_gradient_animated():
    """8. Virtual Gradient: Magma Diffusion Animation"""
    x = np.linspace(-3, 3, 60)
    y = np.linspace(-3, 3, 60)
    X, Y = np.meshgrid(x, y)
    
    Z_cold = 0.2 * np.sin(X) * np.cos(Y)
    R2 = X**2 + Y**2
    
    frames = []
    # Heat Diffusion Logic
    # Sigma (Spread) increases over time
    # Amplitude increases then stabilizes
    
    for i, t in enumerate(np.linspace(0, 1, 40)):
        # Time operator for spread
        sigma = 0.5 + 2.0 * t # Spread grows
        amp = 2.5 * min(1.0, t * 1.5) # Heat rises fast
        
        Heat = amp * np.exp(-R2 / sigma)
        
        # Add "boiling" noise
        noise = 0.1 * np.random.rand(*X.shape) * t
        
        Z_current = Z_cold + Heat + noise
        
        frames.append(go.Frame(
            data=[go.Surface(
                z=Z_current, x=x, y=y, 
                colorscale='Magma', 
                cmin=-1, cmax=4, # Locked range for burning saturation
                opacity=0.95, showscale=False
            )],
            name=f'fr{i}'
        ))

    fig = go.Figure(
        data=[go.Surface(z=Z_cold, x=x, y=y, colorscale='Magma', cmin=-1, cmax=4, opacity=0.95, showscale=False, name='Weight Matrix')],
        layout=go.Layout(
            title="8. Virtual Gradient: Magma Handprint (Heat Diffusion)",
            scene=dict(zaxis_title="Activation"),
            template=TEMPLATE,
            **layout_defaults,
            updatemenus=[dict(
                type="buttons", 
                buttons=[dict(label="INJECT INTENT (IGNITE)", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])],
                x=0.1, y=0.1
            )]
        ),
        frames=frames
    )
    return fig


def main():
    print("Generating refined animations (V3)...")
    figs = [
        create_fig_1_spacetime_animated(),
        create_fig_2_faket(),
        create_fig_3_human_loss(),
        create_fig_4_attention_grad_animated(),
        create_fig_5_evolution(),
        create_fig_6_resonance(),
        create_fig_7_equivalence(),
        create_fig_8_virtual_gradient_animated(),
    ]

    print("Reading original HTML...")
    with open('/home/kusime/Desktop/AI-Learn/truth/new.html', 'r', encoding='utf-8') as f:
        html_content = f.read()

    new_html = html_content
    
    for i, fig in enumerate(figs, 1):
        print(f"Injecting Figure {i}...")
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True, 'displayModeBar': False})
        
        wrapper = f"""
        <div class="plot-container" style="width:100%; height:80vh; margin: 30px 0; border: 1px solid #222; border-radius: 8px; box-shadow: 0 0 20px rgba(0,0,0,0.5);">
            {plot_html}
        </div>
        """
        
        section_marker = f"<h2>{i}."
        ref_box_marker = '<div class="reference-box">'
        section_start = new_html.find(section_marker)
        if section_start == -1: continue
        ref_start = new_html.find(ref_box_marker, section_start)
        
        if ref_start != -1:
            new_html = new_html[:ref_start] + wrapper + new_html[ref_start:]

    output_path = '/home/kusime/Desktop/AI-Learn/truth/new_visualized.html'
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_html)
    print("Done!")

if __name__ == "__main__":
    main()

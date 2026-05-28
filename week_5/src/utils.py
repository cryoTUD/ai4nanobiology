

def interactive_dot_product(u_color="#c2410c", v_color="#0e7490", half_range=2.4):
    """Display an interactive plot of two vectors and their dot product.

    Four vertical sliders control the angle and length of vectors u and v;
    the plot redraws live, showing u, v, their dot product and the angle
    between them.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    U_COL, V_COL, M = u_color, v_color, half_range

    def draw(u_angle, u_len, v_angle, v_len):
        u = np.array([u_len * np.cos(u_angle), u_len * np.sin(u_angle)])
        v = np.array([v_len * np.cos(v_angle), v_len * np.sin(v_angle)])
        d = float(u @ v)
        cos = d / ((np.linalg.norm(u) * np.linalg.norm(v)) or 1)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.annotate("", xy=u, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=U_COL, lw=3))
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=V_COL, lw=3))
        ax.text(*(u * 1.1), "u", color=U_COL, fontsize=16, fontweight="bold")
        ax.text(*(v * 1.1), "v", color=V_COL, fontsize=16, fontweight="bold")

        ax.set_xlim(-M, M); ax.set_ylim(-M, M); ax.set_aspect("equal")
        ax.axhline(0, color="#cfc8b8", lw=1); ax.axvline(0, color="#cfc8b8", lw=1)
        ax.grid(True, color="#ece7da")
        title_text = (r"$\mathbf{u} \cdot \mathbf{v}$ = " + f"{d:.2f}"
                      + "\n" + r"$\theta$ = " + f"{ang:.0f}°")
        ax.set_title(title_text, fontsize=12)
        plt.show()

    ua = widgets.FloatSlider(value=0.6, min=-np.pi, max=np.pi, step=0.01,
                             description="u angle", readout_format=".2f",
                             orientation="vertical")
    ul = widgets.FloatSlider(value=1.2, min=0.1, max=M, step=0.05,
                             description="u length", readout_format=".2f",
                             orientation="vertical")
    va = widgets.FloatSlider(value=-0.4, min=-np.pi, max=np.pi, step=0.01,
                             description="v angle", readout_format=".2f",
                             orientation="vertical")
    vl = widgets.FloatSlider(value=1.0, min=0.1, max=M, step=0.05,
                             description="v length", readout_format=".2f",
                             orientation="vertical")

    out = widgets.interactive_output(
        draw, {"u_angle": ua, "u_len": ul, "v_angle": va, "v_len": vl})

    display(widgets.HBox([ua, ul, va, vl, out]))


def plot_qv_vectors_from_embedding(e1=None, e2=None, W_q=None, W_k=None):
    """Show two input embeddings (3-D) and their Query/Key projections (2-D).

    Left panel : the two raw embeddings x1, x2 as arrows in 3-D embedding space.
    Right panel: their projections q1 = x1 W_q and k2 = x2 W_k as arrows in the
                 shared 2-D head space, where the attention dot product lives.

    All arguments are optional; sensible defaults are used if not given.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3-D projection)

    # ---- defaults ----
    if e1 is None:
        e1 = np.array([1.0, 0.5, -0.8])
    if e2 is None:
        e2 = np.array([-0.4, 1.1, 0.6])
    if W_q is None:
        W_q = np.array([[0.8, 0.2],
                        [-0.3, 0.9],
                        [0.5, -0.4]])      # shape (d=3, d_h=2)
    if W_k is None:
        W_k = np.array([[-0.4, 0.9],
                        [0.7, 0.1],
                        [0.6, 0.3]])       # shape (d=3, d_h=2)

    e1 = np.asarray(e1, float)
    e2 = np.asarray(e2, float)

    # ---- projections: x_i W  (row vector times (d, d_h) matrix) ----
    q1 = e1 @ W_q      # query of token 1
    k2 = e2 @ W_k      # key   of token 2

    X_COL1, X_COL2 = "#15181f", "#b08400"
    Q_COL, K_COL = "#c2410c", "#0e7490"

    fig = plt.figure(figsize=(11, 4.5))

    # ---- left: 3-D embedding space ----
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    for vec, col, name in [(e1, X_COL1, "x1"), (e2, X_COL2, "x2")]:
        ax1.quiver(0, 0, 0, vec[0], vec[1], vec[2],
                   color=col, lw=2.5, arrow_length_ratio=0.12)
        ax1.text(vec[0]*1.12, vec[1]*1.12, vec[2]*1.12, name,
                 color=col, fontsize=13, fontweight="bold")
    L = 1.6
    ax1.set_xlim(-L, L); ax1.set_ylim(-L, L); ax1.set_zlim(-L, L)
    ax1.set_xlabel("dim 1"); ax1.set_ylabel("dim 2"); ax1.set_zlabel("dim 3")
    ax1.set_title("input embeddings  ($\\mathbb{R}^3$)")
    ax1.view_init(elev=18, azim=40)

    # ---- right: 2-D head space (Q, K projections) ----
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.annotate("", xy=q1, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=Q_COL, lw=3))
    ax2.annotate("", xy=k2, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=K_COL, lw=3))
    ax2.text(q1[0]*1.1, q1[1]*1.1, "q1", color=Q_COL, fontsize=14, fontweight="bold")
    ax2.text(k2[0]*1.1, k2[1]*1.1, "k2", color=K_COL, fontsize=14, fontweight="bold")
    M = max(1.0, np.abs(np.r_[q1, k2]).max()) * 1.3
    ax2.set_xlim(-M, M); ax2.set_ylim(-M, M); ax2.set_aspect("equal")
    ax2.axhline(0, color="#cfc8b8", lw=1); ax2.axvline(0, color="#cfc8b8", lw=1)
    ax2.grid(True, color="#ece7da")
    ax2.set_xlabel("head dim 1"); ax2.set_ylabel("head dim 2")
    ax2.set_title("projections  ($\\mathbb{R}^{d_h}$)")

    fig.suptitle("$q_1 = x_1 W_q$      $k_2 = x_2 W_k$", fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_qkv_vectors_from_embedding(e1=None, e2=None, W_q=None, W_k=None, W_v=None):
    """Show two input embeddings (3-D) and ALL their Q/K/V projections (2-D).

    Left panel : the two raw embeddings x1, x2 as arrows in 3-D embedding space.
    Right panel: every projection in the shared 2-D head space --
                 queries q1, q2;  keys k1, k2;  values v1, v2.
                 Each projection is coloured by the embedding it came from
                 (all arrows from x1 share x1's colour, likewise for x2).

    All arguments are optional; sensible defaults are used if not given.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3-D projection)

    # ---- defaults ----
    if e1 is None:
        e1 = np.array([1.0, 0.5, -0.8])
    if e2 is None:
        e2 = np.array([-0.4, 1.1, 0.6])
    if W_q is None:
        W_q = np.array([[0.8, 0.2], [-0.3, 0.9], [0.5, -0.4]])   # (d=3, d_h=2)
    if W_k is None:
        W_k = np.array([[-0.4, 0.9], [0.7, 0.1], [0.6, 0.3]])    # (d=3, d_h=2)
    if W_v is None:
        W_v = np.array([[0.5, -0.6], [0.5, 0.4], [-0.7, 0.8]])   # (d=3, d_v=2)

    e1 = np.asarray(e1, float)
    e2 = np.asarray(e2, float)

    # ---- projections for BOTH tokens: x_i W ----
    q1, q2 = e1 @ W_q, e2 @ W_q
    k1, k2 = e1 @ W_k, e2 @ W_k
    v1, v2 = e1 @ W_v, e2 @ W_v

    # colour by source embedding
    C1, C2 = "#15181f", "#b08400"

    fig = plt.figure(figsize=(11, 4.5))

    # ---- left: 3-D embedding space ----
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    for vec, col, name in [(e1, C1, "x1"), (e2, C2, "x2")]:
        ax1.quiver(0, 0, 0, vec[0], vec[1], vec[2],
                   color=col, lw=2.5, arrow_length_ratio=0.12)
        ax1.text(vec[0]*1.12, vec[1]*1.12, vec[2]*1.12, name,
                 color=col, fontsize=13, fontweight="bold")
    L = 1.6
    ax1.set_xlim(-L, L); ax1.set_ylim(-L, L); ax1.set_zlim(-L, L)
    ax1.set_xlabel("dim 1"); ax1.set_ylabel("dim 2"); ax1.set_zlabel("dim 3")
    ax1.set_title("input embeddings  ($\\mathbb{R}^3$)")
    ax1.view_init(elev=18, azim=40)

    # ---- right: 2-D head space, all six projections, coloured by source ----
    ax2 = fig.add_subplot(1, 2, 2)

    def arrow(p, col, name):
        ax2.annotate("", xy=p, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
        ax2.text(p[0]*1.07, p[1]*1.07, name, color=col,
                 fontsize=12, fontweight="bold")

    for p, name in [(q1, "q1"), (k1, "k1"), (v1, "v1")]:
        arrow(p, C1, name)
    for p, name in [(q2, "q2"), (k2, "k2"), (v2, "v2")]:
        arrow(p, C2, name)

    M = max(1.0, np.abs(np.r_[q1, q2, k1, k2, v1, v2]).max()) * 1.3
    ax2.set_xlim(-M, M); ax2.set_ylim(-M, M); ax2.set_aspect("equal")
    ax2.axhline(0, color="#cfc8b8", lw=1); ax2.axvline(0, color="#cfc8b8", lw=1)
    ax2.grid(True, color="#ece7da")
    ax2.set_xlabel("head dim 1"); ax2.set_ylabel("head dim 2")
    ax2.set_title("projections  ($\\mathbb{R}^{d_h}$)\ncoloured by source embedding")

    fig.suptitle("each token's $q,k,v$ share that token's colour", fontsize=12)
    plt.tight_layout()
    plt.show()

def interactive_qkv_vectors(e1=None, e2=None, W_q=None, W_k=None, W_v=None):
    """Interactive Query/Key/Value projections with per-matrix rotation and scale.

    W_q, W_k and W_v are each shared across BOTH tokens, as in a single attention
    head. Rotating a matrix moves both of its projected vectors together. The
    three current (rotated + scaled) weight matrices are shown side by side next
    to the sliders, and the title shows both directed scores q1.k2 and q2.k1.

    All dependencies are imported inside the function.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import ipywidgets as widgets
    from IPython.display import display

    if e1 is None: e1 = np.array([1.0, 0.5, -0.8])
    if e2 is None: e2 = np.array([-0.4, 1.1, 0.6])
    if W_q is None: W_q = np.array([[0.8, 0.2], [-0.3, 0.9], [0.5, -0.4]])
    if W_k is None: W_k = np.array([[-0.4, 0.9], [0.7, 0.1], [0.6, 0.3]])
    if W_v is None: W_v = np.array([[0.5, -0.6], [0.5, 0.4], [-0.7, 0.8]])
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    W_q0, W_k0, W_v0 = W_q.copy(), W_k.copy(), W_v.copy()

    C1, C2 = "#15181f", "#b08400"
    Q_COL, K_COL, V_COL = "#c2410c", "#0e7490", "#6d28d9"

    def rot(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def mtx_html(W, title, color):
        rows = "".join(
            "<tr>" + "".join(f"<td style='padding:1px 8px;text-align:right'>{v:+.2f}</td>"
                             for v in row) + "</tr>"
            for row in W)
        return (f"<div style='font-family:monospace;font-size:13px;text-align:center'>"
                f"<b style='color:{color}'>{title}</b>"
                f"<table style='border-left:2px solid {color};margin:4px auto 0'>{rows}</table>"
                f"</div>")

    def all_html(Wq, Wk, Wv):
        return (f"<div style='display:flex;gap:24px;align-items:flex-start'>"
                f"{mtx_html(Wq, 'W_q', Q_COL)}"
                f"{mtx_html(Wk, 'W_k', K_COL)}"
                f"{mtx_html(Wv, 'W_v', V_COL)}</div>")

    mat_box = widgets.HTML()

    def draw(q_rot, q_scale, k_rot, k_scale, v_rot, v_scale):
        Wq = q_scale * (W_q0 @ rot(q_rot).T)
        Wk = k_scale * (W_k0 @ rot(k_rot).T)
        Wv = v_scale * (W_v0 @ rot(v_rot).T)

        mat_box.value = all_html(Wq, Wk, Wv)

        q1, q2 = e1 @ Wq, e2 @ Wq
        k1, k2 = e1 @ Wk, e2 @ Wk
        v1, v2 = e1 @ Wv, e2 @ Wv
        s12 = float(q1 @ k2); s21 = float(q2 @ k1)

        fig = plt.figure(figsize=(11, 4.5))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        for vec, col, name in [(e1, C1, "x1"), (e2, C2, "x2")]:
            ax1.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=col, lw=2.5,
                       arrow_length_ratio=0.12)
            ax1.text(vec[0]*1.12, vec[1]*1.12, vec[2]*1.12, name, color=col,
                     fontsize=13, fontweight="bold")
        L = 1.6
        ax1.set_xlim(-L, L); ax1.set_ylim(-L, L); ax1.set_zlim(-L, L)
        ax1.set_xlabel("dim 1"); ax1.set_ylabel("dim 2"); ax1.set_zlabel("dim 3")
        ax1.set_title("input embeddings  ($\\mathbb{R}^3$)")
        ax1.view_init(elev=18, azim=40)

        ax2 = fig.add_subplot(1, 2, 2)
        def arrow(p, col, name):
            ax2.annotate("", xy=p, xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
            ax2.text(p[0]*1.07, p[1]*1.07, name, color=col, fontsize=12,
                     fontweight="bold")
        for p, name in [(q1, "q1"), (k1, "k1"), (v1, "v1")]:
            arrow(p, C1, name)
        for p, name in [(q2, "q2"), (k2, "k2"), (v2, "v2")]:
            arrow(p, C2, name)

        M = max(1.0, np.abs(np.r_[q1, q2, k1, k2, v1, v2]).max()) * 1.3
        ax2.set_xlim(-M, M); ax2.set_ylim(-M, M); ax2.set_aspect("equal")
        ax2.axhline(0, color="#cfc8b8", lw=1); ax2.axvline(0, color="#cfc8b8", lw=1)
        ax2.grid(True, color="#ece7da")
        ax2.set_xlabel("head dim 1"); ax2.set_ylabel("head dim 2")
        ax2.set_title(f"$q_1\\cdot k_2$ = {s12:.2f}      $q_2\\cdot k_1$ = {s21:.2f}")

        fig.suptitle("rotate / scale each shared matrix — "
                     "maximise the overlap of $q_1$ and $k_2$", fontsize=12)
        plt.tight_layout()
        plt.show()

    def rs(desc):
        r = widgets.FloatSlider(value=0.0, min=-np.pi, max=np.pi, step=0.01,
                                description=f"{desc} rot", readout_format=".2f")
        s = widgets.FloatSlider(value=1.0, min=0.2, max=2.0, step=0.05,
                                description=f"{desc} scale", readout_format=".2f")
        return r, s

    qr, qs = rs("Q"); kr, ks = rs("K"); vr, vs = rs("V")

    out = widgets.interactive_output(draw, {
        "q_rot": qr, "q_scale": qs, "k_rot": kr, "k_scale": ks,
        "v_rot": vr, "v_scale": vs})

    sliders = widgets.VBox([
        widgets.HBox([widgets.HTML("<b style='color:#c2410c'>W_q</b>"), qr, qs]),
        widgets.HBox([widgets.HTML("<b style='color:#0e7490'>W_k</b>"), kr, ks]),
        widgets.HBox([widgets.HTML("<b style='color:#6d28d9'>W_v</b>"), vr, vs]),
    ])
    top = widgets.HBox([sliders, mat_box])
    display(top, out)

def interactive_qk_vectors(e1=None, e2=None, W_q=None, W_k=None):
    """Interactive Query/Key projections with per-matrix rotation and scale.

    W_q and W_k are each shared across BOTH tokens, as in a single attention
    head. Rotating W_q moves q1 AND q2 together; rotating W_k moves k1 AND k2
    together. The current (rotated + scaled) weight matrices are printed next to
    the sliders, and the title shows both directed scores q1.k2 and q2.k1.

    All dependencies are imported inside the function.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import ipywidgets as widgets
    from IPython.display import display

    if e1 is None: e1 = np.array([1.0, 0.5, -0.8])
    if e2 is None: e2 = np.array([-0.4, 1.1, 0.6])
    if W_q is None: W_q = np.array([[0.8, 0.2], [-0.3, 0.9], [0.5, -0.4]])
    if W_k is None: W_k = np.array([[-0.4, 0.9], [0.7, 0.1], [0.6, 0.3]])
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    W_q0, W_k0 = W_q.copy(), W_k.copy()

    C1, C2 = "#15181f", "#b08400"
    Q_COL, K_COL = "#c2410c", "#0e7490"

    def rot(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def mtx_html(W, title, color):
        rows = "".join(
            "<tr>" + "".join(f"<td style='padding:1px 8px;text-align:right'>{v:+.2f}</td>"
                             for v in row) + "</tr>"
            for row in W)
        return (f"<div style='font-family:monospace;font-size:13px;text-align:center'>"
                f"<b style='color:{color}'>{title}</b>"
                f"<table style='border-left:2px solid {color};margin:4px auto 0'>{rows}</table>"
                f"</div>")

    def both_html(Wq, Wk):
        return (f"<div style='display:flex;gap:28px;align-items:flex-start'>"
                f"{mtx_html(Wq, 'W_q', Q_COL)}{mtx_html(Wk, 'W_k', K_COL)}</div>")

    mat_box = widgets.HTML()

    def draw(q_rot, q_scale, k_rot, k_scale):
        Wq = q_scale * (W_q0 @ rot(q_rot).T)
        Wk = k_scale * (W_k0 @ rot(k_rot).T)

        # update the printed matrices (side by side)
        mat_box.value = both_html(Wq, Wk)

        q1, q2 = e1 @ Wq, e2 @ Wq
        k1, k2 = e1 @ Wk, e2 @ Wk
        s12 = float(q1 @ k2); s21 = float(q2 @ k1)

        fig = plt.figure(figsize=(11, 4.5))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        for vec, col, name in [(e1, C1, "x1"), (e2, C2, "x2")]:
            ax1.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=col, lw=2.5,
                       arrow_length_ratio=0.12)
            ax1.text(vec[0]*1.12, vec[1]*1.12, vec[2]*1.12, name, color=col,
                     fontsize=13, fontweight="bold")
        L = 1.6
        ax1.set_xlim(-L, L); ax1.set_ylim(-L, L); ax1.set_zlim(-L, L)
        ax1.set_xlabel("dim 1"); ax1.set_ylabel("dim 2"); ax1.set_zlabel("dim 3")
        ax1.set_title("input embeddings  ($\\mathbb{R}^3$)")
        ax1.view_init(elev=18, azim=40)

        ax2 = fig.add_subplot(1, 2, 2)
        def arrow(p, col, name):
            ax2.annotate("", xy=p, xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
            ax2.text(p[0]*1.07, p[1]*1.07, name, color=col, fontsize=12,
                     fontweight="bold")
        for p, name in [(q1, "q1"), (k1, "k1")]:
            arrow(p, C1, name)
        for p, name in [(q2, "q2"), (k2, "k2")]:
            arrow(p, C2, name)

        M = max(1.0, np.abs(np.r_[q1, q2, k1, k2]).max()) * 1.3
        ax2.set_xlim(-M, M); ax2.set_ylim(-M, M); ax2.set_aspect("equal")
        ax2.axhline(0, color="#cfc8b8", lw=1); ax2.axvline(0, color="#cfc8b8", lw=1)
        ax2.grid(True, color="#ece7da")
        ax2.set_xlabel("head dim 1"); ax2.set_ylabel("head dim 2")
        ax2.set_title(f"$q_1\\cdot k_2$ = {s12:.2f}      $q_2\\cdot k_1$ = {s21:.2f}")

        fig.suptitle("rotate / scale each shared matrix — "
                     "maximise the overlap of $q_1$ and $k_2$", fontsize=12)
        plt.tight_layout()
        plt.show()

    def rs(desc):
        r = widgets.FloatSlider(value=0.0, min=-np.pi, max=np.pi, step=0.01,
                                description=f"{desc} rot", readout_format=".2f")
        s = widgets.FloatSlider(value=1.0, min=0.2, max=2.0, step=0.05,
                                description=f"{desc} scale", readout_format=".2f")
        return r, s

    qr, qs = rs("Q"); kr, ks = rs("K")

    out = widgets.interactive_output(draw, {
        "q_rot": qr, "q_scale": qs, "k_rot": kr, "k_scale": ks})

    sliders = widgets.VBox([
        widgets.HBox([widgets.HTML("<b style='color:#c2410c'>W_q</b>"), qr, qs]),
        widgets.HBox([widgets.HTML("<b style='color:#0e7490'>W_k</b>"), kr, ks]),
    ])
    # sliders on the left, live matrices on the right
    top = widgets.HBox([sliders, mat_box])
    display(top, out)

def interactive_attention(e1=None, e2=None, W_q=None, W_k=None, W_v=None):
    """Full single-head attention, interactive.

    Four panels on the bottom row:
      1. input embeddings x1, x2 in 3-D
      2. query / key projections in the 2-D head space (no values)
      3. value space: input values (dashed) and output values (solid),
         where v_i,out = sum_j alpha_ij v_j
      4. heatmap of the softmax attention weights alpha_ij (annotated)

    W_q, W_k, W_v are shared across tokens. Sliders rotate + scale each matrix.
    Scores are scaled by sqrt(d_h) before the softmax, faithfully.

    All dependencies are imported inside the function.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import ipywidgets as widgets
    from IPython.display import display

    if e1 is None: e1 = np.array([1.0, 0.5, -0.8])
    if e2 is None: e2 = np.array([-0.4, 1.1, 0.6])
    if W_q is None: W_q = np.array([[0.8, 0.2], [-0.3, 0.9], [0.5, -0.4]])
    if W_k is None: W_k = np.array([[-0.4, 0.9], [0.7, 0.1], [0.6, 0.3]])
    if W_v is None: W_v = np.array([[0.5, -0.6], [0.5, 0.4], [-0.7, 0.8]])
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    W_q0, W_k0, W_v0 = W_q.copy(), W_k.copy(), W_v.copy()

    C1, C2 = "#15181f", "#b08400"      # token colours
    Q_COL, K_COL, V_COL = "#c2410c", "#0e7490", "#6d28d9"

    def rot(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def mtx_html(W, title, color):
        rows = "".join(
            "<tr>" + "".join(f"<td style='padding:1px 8px;text-align:right'>{v:+.2f}</td>"
                             for v in row) + "</tr>"
            for row in W)
        return (f"<div style='font-family:monospace;font-size:13px;text-align:center'>"
                f"<b style='color:{color}'>{title}</b>"
                f"<table style='border-left:2px solid {color};margin:4px auto 0'>{rows}</table>"
                f"</div>")

    def all_html(Wq, Wk, Wv):
        return (f"<div style='display:flex;gap:24px;align-items:flex-start'>"
                f"{mtx_html(Wq,'W_q',Q_COL)}{mtx_html(Wk,'W_k',K_COL)}"
                f"{mtx_html(Wv,'W_v',V_COL)}</div>")

    mat_box = widgets.HTML()

    def draw(q_rot, q_scale, k_rot, k_scale, v_rot, v_scale):
        Wq = q_scale * (W_q0 @ rot(q_rot).T)
        Wk = k_scale * (W_k0 @ rot(k_rot).T)
        Wv = v_scale * (W_v0 @ rot(v_rot).T)
        mat_box.value = all_html(Wq, Wk, Wv)

        # projections (rows = tokens)
        Q = np.vstack([e1 @ Wq, e2 @ Wq])      # (2, d_h)
        K = np.vstack([e1 @ Wk, e2 @ Wk])
        V = np.vstack([e1 @ Wv, e2 @ Wv])
        d_h = Wq.shape[1]

        # scaled scores and row-softmax
        scores = (Q @ K.T) / np.sqrt(d_h)       # scores[i, j] = q_i . k_j / sqrt(d_h)
        scores = scores - scores.max(axis=1, keepdims=True)
        exps = np.exp(scores)
        A = exps / exps.sum(axis=1, keepdims=True)   # attention weights, rows sum to 1

        # output values: v_i,out = sum_j A_ij v_j
        V_out = A @ V                            # (2, d_h)

        q1, q2 = Q; k1, k2 = K; v1, v2 = V; vo1, vo2 = V_out

        fig = plt.figure(figsize=(15, 3.8))

        # ---- fig 1: input embeddings (3-D) ----
        ax1 = fig.add_subplot(1, 4, 1, projection="3d")
        for vec, col, name in [(e1, C1, "x1"), (e2, C2, "x2")]:
            ax1.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=col, lw=2.5,
                       arrow_length_ratio=0.12)
            ax1.text(vec[0]*1.12, vec[1]*1.12, vec[2]*1.12, name, color=col,
                     fontsize=12, fontweight="bold")
        L = 1.6
        ax1.set_xlim(-L, L); ax1.set_ylim(-L, L); ax1.set_zlim(-L, L)
        ax1.set_xlabel("dim 1"); ax1.set_ylabel("dim 2"); ax1.set_zlabel("dim 3")
        ax1.set_title("input embeddings ($\\mathbb{R}^3$)", fontsize=11)
        ax1.view_init(elev=18, azim=40)

        # ---- fig 2: query / key projections (2-D) ----
        ax2 = fig.add_subplot(1, 4, 2)
        def arrow(ax, p, col, name, dashed=False):
            ax.annotate("", xy=p, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.3,
                                        linestyle="--" if dashed else "-"))
            ax.text(p[0]*1.08, p[1]*1.08, name, color=col, fontsize=11,
                    fontweight="bold")
        arrow(ax2, q1, C1, "q1"); arrow(ax2, k1, C1, "k1")
        arrow(ax2, q2, C2, "q2"); arrow(ax2, k2, C2, "k2")
        Mqk = max(1.0, np.abs(np.r_[Q, K]).max()) * 1.3
        ax2.set_xlim(-Mqk, Mqk); ax2.set_ylim(-Mqk, Mqk); ax2.set_aspect("equal")
        ax2.axhline(0, color="#cfc8b8", lw=1); ax2.axvline(0, color="#cfc8b8", lw=1)
        ax2.grid(True, color="#ece7da")
        ax2.set_title("query / key ($\\mathbb{R}^{d_h}$)", fontsize=11)

        # ---- fig 3: value space, input (dashed) vs output (solid) ----
        ax3 = fig.add_subplot(1, 4, 3)
        arrow(ax3, v1, C1, "v1", dashed=True)
        arrow(ax3, v2, C2, "v2", dashed=True)
        arrow(ax3, vo1, C1, "v1_out")
        arrow(ax3, vo2, C2, "v2_out")
        Mv = max(1.0, np.abs(np.r_[V, V_out]).max()) * 1.3
        ax3.set_xlim(-Mv, Mv); ax3.set_ylim(-Mv, Mv); ax3.set_aspect("equal")
        ax3.axhline(0, color="#cfc8b8", lw=1); ax3.axvline(0, color="#cfc8b8", lw=1)
        ax3.grid(True, color="#ece7da")
        ax3.set_title("values: input (- -) → output (—)", fontsize=11)

        # ---- fig 4: attention heatmap ----
        ax4 = fig.add_subplot(1, 4, 4)
        im = ax4.imshow(A, cmap="magma", vmin=0, vmax=1)
        ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
        ax4.set_xticklabels(["k1", "k2"]); ax4.set_yticklabels(["q1", "q2"])
        ax4.set_xlabel("key (attended to)"); ax4.set_ylabel("query (from)")
        for i in range(2):
            for j in range(2):
                ax4.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                         color="white" if A[i, j] < 0.6 else "black",
                         fontsize=13, fontweight="bold")
        ax4.set_title(r"$\alpha_{ij}$ (rows sum to 1)", fontsize=11)

        plt.tight_layout()
        plt.show()

    def rs(desc):
        r = widgets.FloatSlider(value=0.0, min=-np.pi, max=np.pi, step=0.01,
                                description=f"{desc} rot", readout_format=".2f")
        s = widgets.FloatSlider(value=1.0, min=0.2, max=2.0, step=0.05,
                                description=f"{desc} scale", readout_format=".2f")
        return r, s

    qr, qs = rs("Q"); kr, ks = rs("K"); vr, vs = rs("V")
    out = widgets.interactive_output(draw, {
        "q_rot": qr, "q_scale": qs, "k_rot": kr, "k_scale": ks,
        "v_rot": vr, "v_scale": vs})
    sliders = widgets.VBox([
        widgets.HBox([widgets.HTML("<b style='color:#c2410c'>W_q</b>"), qr, qs]),
        widgets.HBox([widgets.HTML("<b style='color:#0e7490'>W_k</b>"), kr, ks]),
        widgets.HBox([widgets.HTML("<b style='color:#6d28d9'>W_v</b>"), vr, vs]),
    ])
    display(widgets.HBox([sliders, mat_box]), out)

def interactive_attention_with_output(e1=None, e2=None, W_q=None, W_k=None, W_v=None, W_o=None, b_o=None):
    """Full single-head attention with output projection W_O, interactive.

    Four panels:
      1. 3-D embedding space: input embeddings x1,x2 (dashed) and the projected
         attention outputs x1_out,x2_out (solid), where
         x_i,out = ReLU(v_i,out W_O + b).
      2. query / key projections in the 2-D head space.
      3. value space: input values v1,v2 (dashed) and attention outputs
         v1_out,v2_out (solid), where v_i,out = sum_j alpha_ij v_j.
      4. heatmap of softmax attention weights alpha_ij (annotated).

    All weight matrices are shared across tokens. Sliders rotate + scale each.
    Convention everywhere: INPUT = dashed, OUTPUT = solid.
    Scores scaled by sqrt(d_h) before softmax.

    All dependencies imported inside the function.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import ipywidgets as widgets
    from IPython.display import display

    if e1 is None: e1 = np.array([1.0, 0.5, -0.8])
    if e2 is None: e2 = np.array([-0.4, 1.1, 0.6])
    if W_q is None: W_q = np.array([[0.8, 0.2], [-0.3, 0.9], [0.5, -0.4]])
    if W_k is None: W_k = np.array([[-0.4, 0.9], [0.7, 0.1], [0.6, 0.3]])
    if W_v is None: W_v = np.array([[0.5, -0.6], [0.5, 0.4], [-0.7, 0.8]])
    if W_o is None: W_o = np.array([[0.7, -0.2, 0.5], [0.3, 0.8, -0.4]])   # (d_v=2, d=3)
    if b_o is None: b_o = np.array([0.0, 0.0, 0.0])                        # (1, d)
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    W_q0, W_k0, W_v0, W_o0 = W_q.copy(), W_k.copy(), W_v.copy(), W_o.copy()
    b_o = np.asarray(b_o, float)

    C1, C2 = "#15181f", "#b08400"
    Q_COL, K_COL, V_COL, O_COL = "#c2410c", "#0e7490", "#6d28d9", "#0f766e"

    def rot(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def mtx_html(W, title, color, fs=12):
        rows = "".join(
            "<tr>" + "".join(f"<td style='padding:1px 5px;text-align:right'>{v:+.2f}</td>"
                             for v in row) + "</tr>"
            for row in W)
        return (f"<div style='font-family:monospace;font-size:{fs}px;text-align:center'>"
                f"<b style='color:{color}'>{title}</b>"
                f"<table style='border-left:2px solid {color};margin:3px auto 0'>{rows}</table>"
                f"</div>")

    def all_html(Wq, Wk, Wv, Wo):
        return (f"<div style='display:flex;gap:16px;align-items:flex-start'>"
                f"{mtx_html(Wq,'W_q',Q_COL)}{mtx_html(Wk,'W_k',K_COL)}"
                f"{mtx_html(Wv,'W_v',V_COL)}{mtx_html(Wo,'W_O',O_COL)}</div>")

    mat_box = widgets.HTML()

    def draw(q_rot, q_scale, k_rot, k_scale, v_rot, v_scale, o_scale):
        Wq = q_scale * (W_q0 @ rot(q_rot).T)
        Wk = k_scale * (W_k0 @ rot(k_rot).T)
        Wv = v_scale * (W_v0 @ rot(v_rot).T)
        Wo = o_scale * W_o0
        mat_box.value = all_html(Wq, Wk, Wv, Wo)

        Q = np.vstack([e1 @ Wq, e2 @ Wq])
        K = np.vstack([e1 @ Wk, e2 @ Wk])
        V = np.vstack([e1 @ Wv, e2 @ Wv])
        d_h = Wq.shape[1]

        scores = (Q @ K.T) / np.sqrt(d_h)
        scores = scores - scores.max(axis=1, keepdims=True)
        exps = np.exp(scores)
        A = exps / exps.sum(axis=1, keepdims=True)

        V_out = A @ V                                  # (2, d_v)
        X_out = np.maximum(0.0, V_out @ Wo + b_o)      # ReLU, (2, d)

        q1, q2 = Q; k1, k2 = K; v1, v2 = V; vo1, vo2 = V_out
        x1o, x2o = X_out

        fig = plt.figure(figsize=(15, 3.8))

        # ---- fig 1: 3-D embedding space, inputs (dashed) + outputs (solid) ----
        ax1 = fig.add_subplot(1, 4, 1, projection="3d")
        def q3d(vec, col, name, dashed=False):
            ax1.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=col, lw=2.3,
                       arrow_length_ratio=0.12,
                       linestyle="--" if dashed else "-")
            ax1.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, name, color=col,
                     fontsize=11, fontweight="bold")
        q3d(e1, C1, "x1", dashed=True); q3d(e2, C2, "x2", dashed=True)
        q3d(x1o, C1, "x1_out"); q3d(x2o, C2, "x2_out")
        L = max(1.6, np.abs(np.r_[e1, e2, X_out.ravel()]).max() * 1.15)
        ax1.set_xlim(-L, L); ax1.set_ylim(-L, L); ax1.set_zlim(-L, L)
        ax1.set_xlabel("dim 1"); ax1.set_ylabel("dim 2"); ax1.set_zlabel("dim 3")
        ax1.set_title("embeddings: in (- -) / out (—)", fontsize=11)
        ax1.view_init(elev=18, azim=40)

        def arrow(ax, p, col, name, dashed=False):
            ax.annotate("", xy=p, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.3,
                                        linestyle="--" if dashed else "-"))
            ax.text(p[0]*1.08, p[1]*1.08, name, color=col, fontsize=11,
                    fontweight="bold")

        # ---- fig 2: query / key ----
        ax2 = fig.add_subplot(1, 4, 2)
        arrow(ax2, q1, C1, "q1"); arrow(ax2, k1, C1, "k1")
        arrow(ax2, q2, C2, "q2"); arrow(ax2, k2, C2, "k2")
        Mqk = max(1.0, np.abs(np.r_[Q, K]).max()) * 1.3
        ax2.set_xlim(-Mqk, Mqk); ax2.set_ylim(-Mqk, Mqk); ax2.set_aspect("equal")
        ax2.axhline(0, color="#cfc8b8", lw=1); ax2.axvline(0, color="#cfc8b8", lw=1)
        ax2.grid(True, color="#ece7da")
        ax2.set_title("query / key ($\\mathbb{R}^{d_h}$)", fontsize=11)

        # ---- fig 3: values, input (dashed) vs output (solid) ----
        ax3 = fig.add_subplot(1, 4, 3)
        arrow(ax3, v1, C1, "v1", dashed=True); arrow(ax3, v2, C2, "v2", dashed=True)
        arrow(ax3, vo1, C1, "v1_out")
        arrow(ax3, vo2, C2, "v2_out")
        Mv = max(1.0, np.abs(np.r_[V, V_out]).max()) * 1.3
        ax3.set_xlim(-Mv, Mv); ax3.set_ylim(-Mv, Mv); ax3.set_aspect("equal")
        ax3.axhline(0, color="#cfc8b8", lw=1); ax3.axvline(0, color="#cfc8b8", lw=1)
        ax3.grid(True, color="#ece7da")
        ax3.set_title("values: in (- -) / out (—)", fontsize=11)

        # ---- fig 4: attention heatmap ----
        ax4 = fig.add_subplot(1, 4, 4)
        ax4.imshow(A, cmap="magma", vmin=0, vmax=1)
        ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
        ax4.set_xticklabels(["k1", "k2"]); ax4.set_yticklabels(["q1", "q2"])
        ax4.set_xlabel("key (attended to)"); ax4.set_ylabel("query (from)")
        for i in range(2):
            for j in range(2):
                ax4.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                         color="white" if A[i, j] < 0.6 else "black",
                         fontsize=13, fontweight="bold")
        ax4.set_title(r"$\alpha_{ij}$ (rows sum to 1)", fontsize=11)

        plt.tight_layout()
        plt.show()

    def rs(desc):
        r = widgets.FloatSlider(value=0.0, min=-np.pi, max=np.pi, step=0.01,
                                description=f"{desc} rot", readout_format=".2f")
        s = widgets.FloatSlider(value=1.0, min=0.2, max=2.0, step=0.05,
                                description=f"{desc} scale", readout_format=".2f")
        return r, s

    qr, qs = rs("Q"); kr, ks = rs("K"); vr, vs = rs("V")
    osl = widgets.FloatSlider(value=1.0, min=0.2, max=2.0, step=0.05,
                              description="W_O scale", readout_format=".2f")

    out = widgets.interactive_output(draw, {
        "q_rot": qr, "q_scale": qs, "k_rot": kr, "k_scale": ks,
        "v_rot": vr, "v_scale": vs, "o_scale": osl})

    sliders = widgets.VBox([
        widgets.HBox([widgets.HTML("<b style='color:#c2410c'>W_q</b>"), qr, qs]),
        widgets.HBox([widgets.HTML("<b style='color:#0e7490'>W_k</b>"), kr, ks]),
        widgets.HBox([widgets.HTML("<b style='color:#6d28d9'>W_v</b>"), vr, vs]),
        widgets.HBox([widgets.HTML("<b style='color:#0f766e'>W_O</b>"), osl]),
    ])
    display(widgets.HBox([sliders, mat_box]), out)

def search_word_in_proteome(word, context=5, fasta_file="human_proteome.fasta"):
    word = word.upper()

    invalid = [c for c in word if c in "BJOXZ"]
    if invalid:
        print(f"Invalid letters {invalid} — not in the amino acid alphabet.")
        return

    with open(fasta_file, "r") as f:
        fasta_text = f.read()

    entries = fasta_text.strip().split(">")[1:]

    BOLD = "\033[1m"
    RESET = "\033[0m"

    found = []
    for entry in entries:
        lines = entry.strip().split("\n")
        header = lines[0]
        seq = "".join(lines[1:])
        if word in seq:
            acc = header.split("|")[1] if "|" in header else header.split()[0]
            name = header.split("|")[2].split(" OS=")[0] if "|" in header else header
            pos = seq.find(word)
            snippet = f"...{seq[max(0, pos-context):pos]}{BOLD}{word}{RESET}{seq[pos+len(word):pos+len(word)+context]}..."
            found.append((acc, name, pos + 1, snippet))

    if not found:
        print(f"No proteins found containing '{word}'.")
        return

    print(f"\nFound {len(found)} protein(s) containing '{word}':\n")
    for acc, name, pos, snippet in found:
        print(f"  {acc}  {name}")
        print(f"  Position: {pos},  context: {snippet}")
        print(f"  https://www.uniprot.org/uniprotkb/{acc}\n")
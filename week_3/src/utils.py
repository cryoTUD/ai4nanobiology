def display_protein_with_graph(protein_graph):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import networkx as nx
    from graphein.protein.visualisation import plotly_protein_structure_graph

    protein_plot = plotly_protein_structure_graph(
        protein_graph,
        colour_edges_by="kind",
        colour_nodes_by="degree",
        label_node_ids=False,
        plot_title="",
        node_size_multiplier=1
        )

    # Convert to NetworkX graph
    nx_graph = nx.Graph(protein_graph)
    pos_2d = nx.spring_layout(nx_graph, seed=0)

    # Build edge traces for the 2D graph
    edge_x, edge_y = [], []
    for edge in nx_graph.edges():
        x0, y0 = pos_2d[edge[0]]
        x1, y1 = pos_2d[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='all',
        mode='lines'
    )

    # Build node trace for the 2D graph
    node_x = [pos_2d[node][0] for node in nx_graph.nodes()]
    node_y = [pos_2d[node][1] for node in nx_graph.nodes()]
    node_labels = list(nx_graph.nodes())

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',           # was 'markers'
        text=node_labels,              # text shown next to each node
        textposition='top center',     # where the label sits relative to marker
        hovertext=node_labels,         # text shown on hover
        hoverinfo='text',
        marker=dict(size=15, color='lightblue', line=dict(width=1, color='black')),
        textfont=dict(size=8),
    )

    # Create side-by-side subplot
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'xy'}]],
        subplot_titles=("3D Protein Structure", "Graph Representation"),
    )

    # Add 3D structure traces (extracted from plot_1) to left panel
    for trace in protein_plot.data:
        fig.add_trace(trace, row=1, col=1)

    # Add 2D graph traces to right panel
    fig.add_trace(edge_trace, row=1, col=2)
    fig.add_trace(node_trace, row=1, col=2)
    fig.update_scenes(
        xaxis=dict(showgrid=False, showticklabels=False, showbackground=False, title=''),
        yaxis=dict(showgrid=False, showticklabels=False, showbackground=False, title=''),
        zaxis=dict(showgrid=False, showticklabels=False, showbackground=False, title=''),
        row=1, col=1
    )

    fig.update_layout(height=600, width=1200, showlegend=False)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=2)

    fig.show()
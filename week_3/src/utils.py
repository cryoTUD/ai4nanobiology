import numpy as np
import torch
import networkx as nx

from pathlib import Path
from urllib.request import urlretrieve
import gemmi

### PDB tools ###
def show_raw_lines(file_path: str | Path, n: int = 20):
    """
    Show the first non-empty lines of a text-based structure file.
    """
    file_path = Path(file_path)

    print(f"\nRaw preview: {file_path.name}")
    print("-" * 100)

    shown = 0

    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip()

            if not line.strip():
                continue

            print(f"{line_number:>5}: {line}")
            shown += 1

            if shown >= n:
                break


# ------------------------------------------------------------
# PDB-specific inspection
# ------------------------------------------------------------

def show_pdb_atom_records(pdb_file: str | Path, n: int = 12):
    """
    Show raw ATOM/HETATM records from a PDB file.

    PDB files use fixed-width columns, so the meaning of each value
    is determined by its character position in the line.
    """
    import gemmi
    pdb_file = Path(pdb_file)

    shown = 0

    with pdb_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                print(line.rstrip())
                shown += 1

                if shown >= n:
                    break


def parse_pdb_atom_line(line: str) -> dict:
    """
    Parse selected fields from a fixed-width PDB ATOM/HETATM line.
    """
    return {
        "record": line[0:6].strip(),
        "serial": line[6:11].strip(),
        "atom_name": line[12:16].strip(),
        "altloc": line[16:17].strip(),
        "residue_name": line[17:20].strip(),
        "chain_id": line[21:22].strip(),
        "residue_number": line[22:26].strip(),
        "insertion_code": line[26:27].strip(),
        "x": line[30:38].strip(),
        "y": line[38:46].strip(),
        "z": line[46:54].strip(),
        "occupancy": line[54:60].strip(),
        "b_factor": line[60:66].strip(),
        "element": line[76:78].strip(),
    }

# ------------------------------------------------------------
# mmCIF-specific inspection
# ------------------------------------------------------------
def show_mmcif_categories(cif_file: str | Path, max_categories: int = 30):
    """
    Show mmCIF category names by scanning tags in the text representation.

    mmCIF tags usually look like:
        _entry.id
        _cell.length_a
        _atom_site.Cartn_x

    The category is the part before the dot:
        _entry
        _cell
        _atom_site
    """
    cif_file = Path(cif_file)

    doc = gemmi.cif.read_file(str(cif_file))
    block = doc.sole_block()

    categories = set()

    for line in block.as_string().splitlines():
        line = line.strip()

        if line.startswith("_") and "." in line:
            tag = line.split()[0]
            category = tag.split(".")[0]
            categories.add(category)

    categories = sorted(categories)

    print("\nmmCIF categories found in the file")
    print("-" * 100)

    for category in categories[:max_categories]:
        print(category)

    if len(categories) > max_categories:
        print(f"... plus {len(categories) - max_categories} more categories")


def show_mmcif_atom_site_loop(cif_file: str | Path, n: int = 10):
    """
    Show selected columns from the mmCIF _atom_site category.

    The _atom_site category is the mmCIF equivalent of PDB ATOM/HETATM records.
    Unlike PDB, it is stored as a table with named columns.
    """
    cif_file = Path(cif_file)

    doc = gemmi.cif.read_file(str(cif_file))
    block = doc.sole_block()

    atom_site = block.find(
        "_atom_site.",
        [
            "group_PDB",
            "id",
            "type_symbol",
            "label_atom_id",
            "label_comp_id",
            "auth_asym_id",
            "auth_seq_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
            "occupancy",
            "B_iso_or_equiv",
        ],
    )

    headers = [
        "group",
        "id",
        "elem",
        "atom",
        "res",
        "chain",
        "seq",
        "x",
        "y",
        "z",
        "occ",
        "B",
    ]

    print("\nmmCIF _atom_site loop")
    print("-" * 130)
    print(" ".join(f"{h:>10}" for h in headers))
    print("-" * 130)

    for i, row in enumerate(atom_site):
        if i >= n:
            break

        print(" ".join(f"{value:>10}" for value in row))


# ------------------------------------------------------------
# gemmi parsed structure view
# ------------------------------------------------------------

def show_gemmi_structure_summary(file_path: str | Path):
    """
    Show the common gemmi Structure hierarchy.

    This works for both PDB and mmCIF files.
    """
    file_path = Path(file_path)

    structure = gemmi.read_structure(str(file_path), merge_chain_parts=False)

    print(f"\nParsed structure summary: {file_path.name}")
    print("-" * 100)

    print(f"Structure name: {structure.name}")
    print(f"Number of models: {len(structure)}")

    if len(structure) == 0:
        return

    model = structure[0]

    chain_count = len(model)
    residue_count = 0
    atom_count = 0

    for chain in model:
        for residue in chain:
            residue_count += 1
            atom_count += len(residue)

    print(f"Model number: {model.num}")
    print(f"Chains in first model: {chain_count}")
    print(f"Residues in first model: {residue_count}")
    print(f"Atoms in first model: {atom_count}")


def show_gemmi_atom_table(file_path: str | Path, n: int = 10):
    """
    Show atoms using gemmi's parsed Structure / Model / Chain / Residue / Atom hierarchy.
    """
    file_path = Path(file_path)

    structure = gemmi.read_structure(str(file_path), merge_chain_parts=False)

    print(f"\nParsed atom table from gemmi: {file_path.name}")
    print("-" * 120)

    if len(structure) == 0:
        print("No models found.")
        return

    model = structure[0]

    print(
        f"{'model':<6} "
        f"{'chain':<6} "
        f"{'residue':<10} "
        f"{'seqid':<8} "
        f"{'atom':<8} "
        f"{'elem':<6} "
        f"{'x':>10} "
        f"{'y':>10} "
        f"{'z':>10} "
        f"{'occ':>8} "
        f"{'B':>8}"
    )

    shown = 0

    for chain in model:
        for residue in chain:
            for atom in residue:
                print(
                    f"{model.num:<6} "
                    f"{chain.name:<6} "
                    f"{residue.name:<10} "
                    f"{str(residue.seqid):<8} "
                    f"{atom.name:<8} "
                    f"{atom.element.name:<6} "
                    f"{atom.pos.x:>10.3f} "
                    f"{atom.pos.y:>10.3f} "
                    f"{atom.pos.z:>10.3f} "
                    f"{atom.occ:>8.2f} "
                    f"{atom.b_iso:>8.2f}"
                )

                shown += 1

                if shown >= n:
                    return
                
### Protein graph utils ---
def display_graph(graph):
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(graph, seed=0)
    edge_labels = nx.get_edge_attributes(graph, 'edge_attr')
    edge_labels = {edge: f"{attr.item():.2f}" for edge, attr in edge_labels.items()}
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    plt.show()
    
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

def show_random_elements_in_dictionary(input_dict, n_items, seed=None):
    import random
    if seed is not None:
        random.seed(seed)   
    keys = list(input_dict.keys())
    random_keys = random.sample(keys, n_items)
    for key in random_keys:
        print(f"Key: {key}, Value: {input_dict[key]}")


# -- GNN utils ---

def return_protein_graph_for_pdb_path(pdb_path, params=None):
    import os
    import graphein.protein as gp
    from graphein.protein.graphs import construct_graph

    if params is None:
        params = {
            "granularity": "CA",
            "edge_construction_functions": [gp.add_peptide_bonds],
            "edge_distance_cutoff": 6.0,
        }

    config = gp.ProteinGraphConfig(**params)

    protein_graph = construct_graph(config=config, path=pdb_path)
    return protein_graph

def get_distance_matrix(coords):
    diff_tensor = np.expand_dims(coords, axis=1) - np.expand_dims(coords, axis=0)
    distance_matrix = np.sqrt(np.sum(np.power(diff_tensor, 2), axis=-1))
    return distance_matrix

def convert_pdb_path_to_graph(pdb_path, distance_threshold=6.0, contain_b_factor=True):
    from biopandas.pdb import PandasPdb
    atom_df = PandasPdb().read_pdb(pdb_path)
    atom_df = atom_df.df['ATOM']
    residue_df = atom_df.groupby('residue_number', as_index=False)[['x_coord', 'y_coord', 'z_coord', 'b_factor']].mean().sort_values('residue_number')
    coords = residue_df[['x_coord', 'y_coord', 'z_coord']].values
    distance_matrix = get_distance_matrix(coords)
    adj = distance_matrix < distance_threshold
    u, v = np.nonzero(adj)
    u, v = torch.from_numpy(u), torch.from_numpy(v)
    graph = nx.Graph()

    # Add nodes with coordinates and b_factor included in the node feature matrix 'x'
    for i, row in residue_df.iterrows():
        node_features = [row['x_coord'], row['y_coord'], row['z_coord']]
        if contain_b_factor:
            node_features.append(row['b_factor'])  # Append b_factor to the features
        graph.add_node(i, x=torch.tensor(node_features, dtype=torch.float))

    # Add edges based on the adjacency matrix
    for src, dst in zip(u.numpy(), v.numpy()):
        distance = distance_matrix[src, dst]
        graph.add_edge(src, dst, edge_attr=torch.tensor([distance], dtype=torch.float))

    return graph


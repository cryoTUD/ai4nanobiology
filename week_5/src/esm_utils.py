def tokenise_pdb_for_esm(pdb_path, model_name="facebook/esm2_t6_8M_UR50D",
                         chain=None, return_sequence=False):
    """Parse a PDB/mmCIF file -> amino-acid sequence -> ESM-2 tokenised input.

    ESM-2 tokenises a residue sequence string (not a 3-D structure), so this
    reads the sequence from the structure's ATOM records and tokenises it.

    Parameters
    ----------
    pdb_path   : path to a .pdb or .cif/.mmcif file
    model_name : ESM-2 checkpoint whose tokenizer to use
    chain      : chain ID to extract (e.g. "A"); if None, the first chain is used
    return_sequence : also include the raw one-letter sequence in the output

    Returns
    -------
    dict with:
      input_ids      : (1, L+2) tensor  (L = #residues; +2 for <cls>/<eos>)
      attention_mask : (1, L+2) tensor
      tokens         : list[str] of token strings, including <cls>/<eos>
      chain          : the chain ID actually used
      sequence       : (if return_sequence) the one-letter amino-acid string

    Notes
    -----
    - Sequence comes from ATOM records, so unresolved residues are absent
      (lengths then match a contact map built from the same structure).
    - Non-standard residues are mapped to "X".
    - <cls>/<eos> are prepended/appended; strip positions 0 and -1 before
      aligning attention to a residue-level contact map.
    """
    import os
    os.environ["HF_HOME"] = "/projects/nb4170/esm_models"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.Data.IUPACData import protein_letters_3to1_extended as _3to1
    from transformers import AutoTokenizer

    # ---- pick parser by extension ----
    ext = os.path.splitext(str(pdb_path))[1].lower()
    if ext in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = next(structure.get_models())          # first model only

    # ---- choose chain ----
    chains = list(model.get_chains())
    if not chains:
        raise ValueError("No chains found in the structure.")
    if chain is None:
        chosen = chains[0]
    else:
        match = [c for c in chains if c.id == chain]
        if not match:
            raise ValueError(f"Chain {chain!r} not found (have: "
                             f"{[c.id for c in chains]}).")
        chosen = match[0]

    # ---- extract one-letter sequence from amino-acid residues, in order ----
    seq_chars = []
    for residue in chosen.get_residues():
        if not is_aa(residue, standard=False):
            continue                              # skip water / ligands / ions
        resname = residue.get_resname().strip().capitalize()  # e.g. "ALA"->"Ala"
        one = _3to1.get(resname, "X")
        seq_chars.append(one if len(one) == 1 else "X")
    sequence = "".join(seq_chars)
    if not sequence:
        raise ValueError(f"No amino-acid residues found in chain {chosen.id!r}.")

    # ---- tokenise for ESM-2 ----
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    enc = tokenizer(sequence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    out = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "tokens": tokens,
        "chain": chosen.id,
    }
    if return_sequence:
        out["sequence"] = sequence
    return out


def pdb_to_contact_map(pdb_path, distance=6.0, chain=None, atom="CB",
                       return_sequence=False):
    """Compute a residue contact map from a structure file.

    Two residues are 'in contact' if the distance between their representative
    atoms is < `distance` Angstroms. This matches the residue ordering produced
    by `tokenise_pdb_for_esm` (ATOM-record residues of one chain, in order), so
    the contact map can be overlaid on ESM-2 attention directly.

    Parameters
    ----------
    pdb_path : path to a .pdb or .cif/.mmcif file
    distance : contact threshold in Angstroms (default 6.0)
    chain    : chain ID (e.g. "A"); if None, the first chain is used
    atom     : representative atom per residue -- "CB" (beta carbon, standard for
               contact maps; falls back to CA for glycine) or "CA"
    return_sequence : also return the one-letter sequence

    Returns
    -------
    dict with:
      contacts  : (L, L) bool numpy array, True where residues are in contact
      distances : (L, L) float numpy array of pairwise distances (Angstrom)
      chain     : chain ID used
      sequence  : (if return_sequence) one-letter amino-acid string

    Note
    ----
    L here counts only resolved amino-acid residues -- it does NOT include the
    <cls>/<eos> tokens ESM adds. Strip those two attention positions before
    overlaying.
    """
    import os
    import numpy as np
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.Data.IUPACData import protein_letters_3to1_extended as _3to1

    ext = os.path.splitext(str(pdb_path))[1].lower()
    parser = MMCIFParser(QUIET=True) if ext in (".cif", ".mmcif") else PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = next(structure.get_models())

    chains = list(model.get_chains())
    if not chains:
        raise ValueError("No chains found in the structure.")
    if chain is None:
        chosen = chains[0]
    else:
        match = [c for c in chains if c.id == chain]
        if not match:
            raise ValueError(f"Chain {chain!r} not found (have: {[c.id for c in chains]}).")
        chosen = match[0]

    # representative coordinate per amino-acid residue, in chain order
    coords, seq_chars = [], []
    for residue in chosen.get_residues():
        if not is_aa(residue, standard=False):
            continue                                  # skip water / ligands / ions
        # pick representative atom: CB, fall back to CA (e.g. glycine has no CB)
        if atom == "CB" and "CB" in residue:
            rep = residue["CB"]
        elif "CA" in residue:
            rep = residue["CA"]
        else:
            continue                                  # residue with no usable atom
        coords.append(rep.get_coord())
        resname = residue.get_resname().strip().capitalize()
        one = _3to1.get(resname, "X")
        seq_chars.append(one if len(one) == 1 else "X")

    if not coords:
        raise ValueError(f"No usable residues found in chain {chosen.id!r}.")

    coords = np.asarray(coords, dtype=float)          # (L, 3)
    # pairwise Euclidean distances via broadcasting
    diff = coords[:, None, :] - coords[None, :, :]    # (L, L, 3)
    distances = np.sqrt((diff ** 2).sum(axis=-1))     # (L, L)
    contacts = distances < distance                    # bool (L, L); diagonal True

    result = {"contacts": contacts, "distances": distances, "chain": chosen.id}
    if return_sequence:
        result["sequence"] = "".join(seq_chars)
    return result


def display_heatmap(array, tokenised_pdb=None, cmap="turbo"):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(array, cmap=cmap)
    input_tokens = tokenised_pdb['tokens']
    labels_str = [f"{input_tokens[i]}_{i}" for i in np.arange(1,len(input_tokens)-1, step=5)]
    plt.xticks(ticks=np.arange(len(input_tokens)-2, step=5), labels=labels_str, rotation=90);
    plt.yticks(ticks=np.arange(len(input_tokens)-2, step=5), labels=labels_str, rotation=0);
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")


def display_heatmaps(list_of_arrays, tokenised_pdb=None, cmap="turbo", titles=None):
    import matplotlib.pyplot as plt
    n = len(list_of_arrays)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    for i, array in enumerate(list_of_arrays):
        plt.sca(axes[i])
        display_heatmap(array, tokenised_pdb=tokenised_pdb, cmap=cmap)
        if titles is not None:
            axes[i].set_title(titles[i])
    
    fig.tight_layout()
    
    
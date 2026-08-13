#!/usr/bin/env python3
"""Plot chemical-space characteristics of the final virtual-screening benchmark.

Fragments are excluded from all analyses. The script reproduces four figures:
    vs_representative_pairs.png
    vs_nearest_neighbour_similarity.png
    vs_descriptor_distributions.png
    vs_umap.png

Run from the repository root with:
    python plotting/scripts/plot_virtual_screening_characterisation.py
"""

from __future__ import annotations

import argparse
import warnings
from io import BytesIO
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdDepictor,
    rdFMCS,
    rdMolDescriptors,
)
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_BENCHMARK = (
    REPO_ROOT / "virtual_screening" / "benchmark" / "virtual_screening_benchmark.csv"
)
DEFAULT_FIGURES_DIR = REPO_ROOT / "plotting" / "figures"

ECFP_RADIUS = 2
ECFP_BITS = 2048
UMAP_N_NEIGHBOURS = 30
UMAP_MIN_DIST = 0.10
RANDOM_STATE = 1
N_REPRESENTATIVE_PAIRS = 4
PAIR_MIN_SIMILARITY = 0.50
PAIR_MAX_SIMILARITY = 1.00

VIOLIN_ALPHA = 0.45
HIGHLIGHT_ALPHA = 0.25
PLOT_COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

RDLogger.DisableLog("rdApp.*")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot chemical-space characteristics of the final VS benchmark."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Final virtual-screening benchmark CSV.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Output directory for PNG figures.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=REPO_ROOT / "plotting" / "tables",
        help="Output directory for summary tables.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def parse_bool_column(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    values = series.astype("string").str.strip().str.lower()
    parsed = values.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    ).astype("boolean")

    if parsed.isna().any():
        examples = series.loc[parsed.isna()].astype(str).unique()[:10]
        raise ValueError(
            f"Missing or unrecognised boolean values in {name!r}: "
            + ", ".join(examples)
        )

    return parsed.astype(bool)


def read_benchmark(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"Name", "SMILES", "is_binder"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Virtual-screening benchmark is missing columns: "
            + ", ".join(sorted(missing))
        )

    df = df[["Name", "SMILES", "is_binder"]].copy()
    df["Name"] = df["Name"].astype("string").str.strip()
    df["SMILES"] = df["SMILES"].astype("string").str.strip()
    df["is_binder"] = parse_bool_column(df["is_binder"], "is_binder")

    if df["Name"].isna().any() or df["Name"].eq("").any():
        raise ValueError("Benchmark contains missing compound names.")
    if df["SMILES"].isna().any() or df["SMILES"].eq("").any():
        raise ValueError("Benchmark contains missing SMILES.")

    if df["is_binder"].nunique() != 2:
        raise ValueError(
            "Virtual-screening benchmark must contain both binders "
            "and suspected non-binders."
        )

    return df.reset_index(drop=True)


def mol_from_smiles(smiles: object) -> Chem.Mol | None:
    if pd.isna(smiles):
        return None
    return Chem.MolFromSmiles(str(smiles))


def build_ecfp4(
    smiles: pd.Series,
) -> tuple[pd.Series, list, np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=ECFP_RADIUS,
        fpSize=ECFP_BITS,
        includeChirality=True,
    )

    valid = np.ones(len(smiles), dtype=bool)
    fps: list = []
    bit_rows: list[np.ndarray] = []

    for i, smi in enumerate(smiles):
        mol = mol_from_smiles(smi)
        if mol is None:
            valid[i] = False
            fps.append(None)
            bit_rows.append(np.zeros(ECFP_BITS, dtype=np.uint8))
            continue

        fp = generator.GetFingerprint(mol)
        arr = np.zeros(ECFP_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(fp)
        bit_rows.append(arr)

    return pd.Series(valid, index=smiles.index), fps, np.vstack(bit_rows)


def compute_tanimoto_matrix(fps: list) -> np.ndarray:
    n = len(fps)
    similarity = np.eye(n, dtype=np.float32)

    for i, fp in enumerate(fps[:-1]):
        values = DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :])
        similarity[i, i + 1 :] = values
        similarity[i + 1 :, i] = values

    return similarity


def nearest_neighbour_metrics(
    df: pd.DataFrame,
    similarity: np.ndarray,
) -> pd.DataFrame:
    labels = df["is_binder"].to_numpy(dtype=bool)
    names = df["Name"].astype(str).to_numpy()
    smiles = df["SMILES"].astype(str).to_numpy()

    binder_idx = np.flatnonzero(labels)
    nonbinder_idx = np.flatnonzero(~labels)
    rows: list[dict[str, object]] = []

    for i in range(len(df)):
        same_pool = binder_idx if labels[i] else nonbinder_idx
        opposite_pool = nonbinder_idx if labels[i] else binder_idx
        same_pool = same_pool[same_pool != i]

        j_same = (
            int(same_pool[np.argmax(similarity[i, same_pool])])
            if len(same_pool)
            else None
        )
        j_opp = (
            int(opposite_pool[np.argmax(similarity[i, opposite_pool])])
            if len(opposite_pool)
            else None
        )

        same_name = names[j_same] if j_same is not None else ""
        same_smiles = smiles[j_same] if j_same is not None else ""
        same_similarity = float(similarity[i, j_same]) if j_same is not None else np.nan

        opposite_name = names[j_opp] if j_opp is not None else ""
        opposite_smiles = smiles[j_opp] if j_opp is not None else ""
        opposite_similarity = (
            float(similarity[i, j_opp]) if j_opp is not None else np.nan
        )

        if labels[i]:
            nearest_binder_name = same_name
            nearest_binder_smiles = same_smiles
            nearest_binder_similarity = same_similarity
            nearest_nonbinder_name = opposite_name
            nearest_nonbinder_smiles = opposite_smiles
            nearest_nonbinder_similarity = opposite_similarity
        else:
            nearest_binder_name = opposite_name
            nearest_binder_smiles = opposite_smiles
            nearest_binder_similarity = opposite_similarity
            nearest_nonbinder_name = same_name
            nearest_nonbinder_smiles = same_smiles
            nearest_nonbinder_similarity = same_similarity

        rows.append(
            {
                "Name": names[i],
                "SMILES": smiles[i],
                "is_binder": bool(labels[i]),
                "nearest_binder_name": nearest_binder_name,
                "nearest_binder_smiles": nearest_binder_smiles,
                "nearest_binder_similarity": nearest_binder_similarity,
                "nearest_nonbinder_name": nearest_nonbinder_name,
                "nearest_nonbinder_smiles": nearest_nonbinder_smiles,
                "nearest_nonbinder_similarity": nearest_nonbinder_similarity,
            }
        )

    return pd.DataFrame(rows)


def build_characterisation_summary(
    descriptors: pd.DataFrame,
    nn: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise the main VS benchmark characterisation metrics."""

    def summarise(series: pd.Series) -> tuple[float, float, float]:
        values = pd.to_numeric(series, errors="coerce").dropna()
        return (
            float(values.median()),
            float(values.quantile(0.25)),
            float(values.quantile(0.75)),
        )

    rows: list[dict[str, object]] = []

    for is_binder, label in [
        (True, "Binder"),
        (False, "Suspected non-binder"),
    ]:
        desc = descriptors.loc[descriptors["is_binder"].eq(is_binder)]
        neighbours = nn.loc[nn["is_binder"].eq(is_binder)]

        if is_binder:
            same_class = neighbours["nearest_binder_similarity"]
            opposite_class = neighbours["nearest_nonbinder_similarity"]
        else:
            same_class = neighbours["nearest_nonbinder_similarity"]
            opposite_class = neighbours["nearest_binder_similarity"]

        mw = summarise(desc["Molecular Weight"])
        clogp = summarise(desc["cLogP"])
        tpsa = summarise(desc["Topological Polar Surface Area"])
        hac = summarise(desc["Heavy Atom Count"])
        same = summarise(same_class)
        opposite = summarise(opposite_class)

        rows.append(
            {
                "group": label,
                "n_compounds": len(desc),
                "molecular_weight_median": mw[0],
                "molecular_weight_q25": mw[1],
                "molecular_weight_q75": mw[2],
                "clogp_median": clogp[0],
                "clogp_q25": clogp[1],
                "clogp_q75": clogp[2],
                "tpsa_median": tpsa[0],
                "tpsa_q25": tpsa[1],
                "tpsa_q75": tpsa[2],
                "heavy_atom_count_median": hac[0],
                "heavy_atom_count_q25": hac[1],
                "heavy_atom_count_q75": hac[2],
                "nearest_same_class_tanimoto_median": same[0],
                "nearest_same_class_tanimoto_q25": same[1],
                "nearest_same_class_tanimoto_q75": same[2],
                "nearest_opposite_class_tanimoto_median": opposite[0],
                "nearest_opposite_class_tanimoto_q25": opposite[1],
                "nearest_opposite_class_tanimoto_q75": opposite[2],
            }
        )

    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_nearest_neighbour_similarity(
    nn: pd.DataFrame,
    save_path: Path,
) -> None:
    data = [
        nn.loc[nn["is_binder"], "nearest_nonbinder_similarity"].dropna().to_numpy(),
        nn.loc[nn["is_binder"], "nearest_binder_similarity"].dropna().to_numpy(),
        nn.loc[~nn["is_binder"], "nearest_binder_similarity"].dropna().to_numpy(),
        nn.loc[~nn["is_binder"], "nearest_nonbinder_similarity"].dropna().to_numpy(),
    ]
    labels = [
        "Binder\nnearest non-binder",
        "Binder\nnearest binder",
        "Non-binder\nnearest binder",
        "Non-binder\nnearest non-binder",
    ]

    positions = np.arange(1, 5)
    rng = np.random.default_rng(1)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, colour in zip(parts["bodies"], PLOT_COLOURS[:4]):
        body.set_facecolor(colour)
        body.set_edgecolor("black")
        body.set_linewidth(0.7)
        body.set_alpha(VIOLIN_ALPHA)

    ax.boxplot(
        data,
        positions=positions,
        widths=0.24,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
        boxprops={
            "facecolor": (1.0, 1.0, 1.0, 0.25),
            "edgecolor": "black",
            "linewidth": 0.85,
        },
        whiskerprops={"color": "black", "linewidth": 0.85},
        capprops={"color": "black", "linewidth": 0.85},
    )

    for x, values in zip(positions, data):
        sample = (
            values if len(values) <= 350 else rng.choice(values, 350, replace=False)
        )
        ax.scatter(
            rng.normal(x, 0.035, len(sample)),
            sample,
            s=6,
            alpha=0.20,
            c="black",
            linewidths=0,
            rasterized=True,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Nearest-neighbour ECFP4 Tanimoto similarity")
    ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, save_path)


def molecular_descriptor_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for row in df.itertuples(index=False):
        mol = mol_from_smiles(row.SMILES)
        if mol is None:
            continue

        rows.append(
            {
                "Name": row.Name,
                "is_binder": bool(row.is_binder),
                "Molecular Weight": float(Descriptors.MolWt(mol)),
                "Heavy Atom Count": int(mol.GetNumHeavyAtoms()),
                "cLogP": float(Crippen.MolLogP(mol)),
                "Topological Polar Surface Area": float(rdMolDescriptors.CalcTPSA(mol)),
                "Hydrogen Bond Donors": int(Lipinski.NumHDonors(mol)),
                "Hydrogen Bond Acceptors": int(Lipinski.NumHAcceptors(mol)),
                "Rotatable Bonds": int(Lipinski.NumRotatableBonds(mol)),
                "Aromatic Rings": int(Lipinski.NumAromaticRings(mol)),
                "Fraction sp3 Carbons": float(rdMolDescriptors.CalcFractionCSP3(mol)),
            }
        )

    return pd.DataFrame(rows)


def plot_descriptor_distributions(
    descriptors: pd.DataFrame,
    save_path: Path,
) -> None:
    properties = [
        "Molecular Weight",
        "Heavy Atom Count",
        "cLogP",
        "Topological Polar Surface Area",
        "Hydrogen Bond Donors",
        "Hydrogen Bond Acceptors",
        "Rotatable Bonds",
        "Aromatic Rings",
        "Fraction sp3 Carbons",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12.0, 10.2))
    rng = np.random.default_rng(3)

    for ax, prop in zip(axes.flat, properties):
        data = [
            descriptors.loc[descriptors["is_binder"], prop].to_numpy(dtype=float),
            descriptors.loc[~descriptors["is_binder"], prop].to_numpy(dtype=float),
        ]

        parts = ax.violinplot(
            data,
            positions=[1, 2],
            widths=0.78,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, colour in zip(parts["bodies"], PLOT_COLOURS[:2]):
            body.set_facecolor(colour)
            body.set_edgecolor("black")
            body.set_linewidth(0.7)
            body.set_alpha(VIOLIN_ALPHA)

        ax.boxplot(
            data,
            positions=[1, 2],
            widths=0.24,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.1},
            boxprops={
                "facecolor": (1.0, 1.0, 1.0, 0.25),
                "edgecolor": "black",
                "linewidth": 0.85,
            },
            whiskerprops={"color": "black", "linewidth": 0.85},
            capprops={"color": "black", "linewidth": 0.85},
        )

        for x, values in enumerate(data, start=1):
            sample = (
                values if len(values) <= 350 else rng.choice(values, 350, replace=False)
            )
            ax.scatter(
                rng.normal(x, 0.035, len(sample)),
                sample,
                s=6,
                alpha=0.20,
                c="black",
                linewidths=0,
                rasterized=True,
            )

        ax.set_title(prop, fontsize=14, pad=6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Binders", "Non-binders"], fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)

    fig.tight_layout()
    save_figure(fig, save_path)


def select_representative_pairs(nn: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        nn.loc[nn["is_binder"]].dropna(subset=["nearest_nonbinder_similarity"]).copy()
    )

    pairs = pairs.loc[
        pairs["nearest_nonbinder_similarity"].between(
            PAIR_MIN_SIMILARITY,
            PAIR_MAX_SIMILARITY,
        )
    ]

    pairs = pairs.sort_values(
        ["nearest_nonbinder_similarity", "Name", "nearest_nonbinder_name"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    if pairs.empty:
        return pairs

    indices = (
        np.linspace(
            0,
            len(pairs) - 1,
            min(N_REPRESENTATIVE_PAIRS, len(pairs)),
        )
        .round()
        .astype(int)
    )

    return (
        pairs.iloc[indices]
        .drop_duplicates(["SMILES", "nearest_nonbinder_smiles"])
        .head(N_REPRESENTATIVE_PAIRS)
        .rename(columns={"Name": "binder_name", "SMILES": "binder_smiles"})
    )


def _highlight_colour_map(
    indices: list[int] | tuple[int, ...],
) -> dict[int, tuple[float, float, float, float]]:
    colour = (0.90, 0.0, 0.0, HIGHLIGHT_ALPHA)
    return {int(index): colour for index in indices}


def _matched_substructure_atoms(
    mol: Chem.Mol,
    query: Chem.Mol | None,
) -> tuple[int, ...]:
    if query is None:
        return ()
    match = mol.GetSubstructMatch(query)
    return tuple(match) if match else ()


def _draw_molecule(
    mol: Chem.Mol,
    highlight_atoms: list[int],
    highlight_bonds: list[int],
    size: tuple[int, int] = (420, 230),
) -> Image.Image:
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    options.padding = 0.06
    options.highlightRadius = 0.20

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=_highlight_colour_map(highlight_atoms),
        highlightBondColors=_highlight_colour_map(highlight_bonds),
    )
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")


def draw_representative_pairs(
    pairs: pd.DataFrame,
    save_path: Path,
) -> None:
    panels: list[dict[str, object]] = []

    for row in pairs.itertuples(index=False):
        binder = mol_from_smiles(row.binder_smiles)
        nonbinder = mol_from_smiles(row.nearest_nonbinder_smiles)
        if binder is None or nonbinder is None:
            continue

        core = None
        try:
            mcs = rdFMCS.FindMCS(
                [binder, nonbinder],
                timeout=5,
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
                matchValences=False,
            )
            if mcs.smartsString:
                core = Chem.MolFromSmarts(mcs.smartsString)
        except Exception:
            core = None

        try:
            if (
                core is not None
                and binder.HasSubstructMatch(core)
                and nonbinder.HasSubstructMatch(core)
            ):
                rdDepictor.Compute2DCoords(core)
                rdDepictor.GenerateDepictionMatching2DStructure(
                    binder, core, acceptFailure=True
                )
                rdDepictor.GenerateDepictionMatching2DStructure(
                    nonbinder, core, acceptFailure=True
                )
            else:
                rdDepictor.Compute2DCoords(binder)
                rdDepictor.GenerateDepictionMatching2DStructure(
                    nonbinder, binder, acceptFailure=True
                )
        except Exception:
            rdDepictor.Compute2DCoords(binder)
            rdDepictor.Compute2DCoords(nonbinder)

        binder_atoms = _matched_substructure_atoms(binder, core)
        nonbinder_atoms = _matched_substructure_atoms(nonbinder, core)

        binder_bonds = [
            bond.GetIdx()
            for bond in binder.GetBonds()
            if bond.GetBeginAtomIdx() in binder_atoms
            and bond.GetEndAtomIdx() in binder_atoms
        ]
        nonbinder_bonds = [
            bond.GetIdx()
            for bond in nonbinder.GetBonds()
            if bond.GetBeginAtomIdx() in nonbinder_atoms
            and bond.GetEndAtomIdx() in nonbinder_atoms
        ]

        panels.extend(
            [
                {
                    "mol": binder,
                    "legend": f"Binder\n{row.binder_name}",
                    "atoms": list(binder_atoms),
                    "bonds": binder_bonds,
                },
                {
                    "mol": nonbinder,
                    "legend": (
                        f"Nearest non-binder\n{row.nearest_nonbinder_name}\n"
                        f"ECFP4 Tanimoto={row.nearest_nonbinder_similarity:.2f}"
                    ),
                    "atoms": list(nonbinder_atoms),
                    "bonds": nonbinder_bonds,
                },
            ]
        )

    if not panels:
        raise ValueError("No representative binder/non-binder pairs could be drawn.")

    n_cols = 2
    n_rows = ceil(len(panels) / n_cols)
    mol_w, mol_h = 420, 230
    cell_w, cell_h = 440, 325

    canvas = Image.new("RGB", (n_cols * cell_w, n_rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for idx, panel in enumerate(panels):
        row_idx, col_idx = divmod(idx, n_cols)
        x0 = col_idx * cell_w
        y0 = row_idx * cell_h

        mol_image = _draw_molecule(
            panel["mol"],
            panel["atoms"],
            panel["bonds"],
            size=(mol_w, mol_h),
        )
        canvas.paste(mol_image, (x0 + (cell_w - mol_w) // 2, y0))

        bbox = draw.multiline_textbbox(
            (0, 0),
            panel["legend"],
            font=font,
            spacing=6,
            align="center",
        )
        width = bbox[2] - bbox[0]

        draw.multiline_text(
            (x0 + (cell_w - width) // 2, y0 + mol_h + 6),
            panel["legend"],
            fill="black",
            font=font,
            spacing=6,
            align="center",
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def compute_umap(bit_matrix: np.ndarray) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP requires 'umap-learn'. Install it in the repository environment."
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        metric="jaccard",
        n_neighbors=UMAP_N_NEIGHBOURS,
        min_dist=UMAP_MIN_DIST,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    # UMAP warns that inverse_transform is unavailable for Jaccard distance.
    # We never call inverse_transform, so suppress only that specific warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="gradient function is not yet implemented for jaccard distance metric.*",
            category=UserWarning,
        )
        return reducer.fit_transform(bit_matrix.astype(bool))


def plot_umap(
    df: pd.DataFrame,
    coordinates: np.ndarray,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.4))

    for is_binder, label, colour in [
        (False, "Non-binder", PLOT_COLOURS[1]),
        (True, "Binder", PLOT_COLOURS[0]),
    ]:
        mask = df["is_binder"].eq(is_binder).to_numpy()
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=14,
            c=colour,
            alpha=0.45,
            label=f"{label} ($n={int(mask.sum())}$)",
            linewidths=0.22,
            edgecolors="black",
            rasterized=True,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(frameon=False, loc="best", handletextpad=0.3, borderaxespad=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, save_path)


def generate_virtual_screening_characterisation(
    benchmark_path: Path,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    benchmark_path = require_file(
        benchmark_path,
        "Virtual-screening benchmark",
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = read_benchmark(benchmark_path)

    valid, fps_all, bit_matrix_all = build_ecfp4(df["SMILES"])
    if not valid.all():
        invalid_names = df.loc[~valid, "Name"].astype(str).tolist()
        raise ValueError(
            "Invalid SMILES in virtual-screening benchmark: "
            + ", ".join(invalid_names[:20])
        )

    fps = [fp for fp in fps_all if fp is not None]
    bit_matrix = bit_matrix_all[valid.to_numpy()]

    similarity = compute_tanimoto_matrix(fps)
    nn = nearest_neighbour_metrics(df, similarity)

    plot_nearest_neighbour_similarity(
        nn,
        figures_dir / "vs_nearest_neighbour_similarity.png",
    )

    descriptors = molecular_descriptor_table(df)

    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = build_characterisation_summary(
        descriptors=descriptors,
        nn=nn,
    )

    summary.to_csv(
        tables_dir / "vs_characterisation_summary.csv",
        index=False,
        float_format="%.3f",
    )

    plot_descriptor_distributions(
        descriptors,
        figures_dir / "vs_descriptor_distributions.png",
    )

    pairs = select_representative_pairs(nn)
    draw_representative_pairs(
        pairs,
        figures_dir / "vs_representative_pairs.png",
    )

    print("Computing UMAP..")
    coordinates = compute_umap(bit_matrix)
    plot_umap(
        df,
        coordinates,
        figures_dir / "vs_umap.png",
    )

    print(
        "VS characterisation: "
        f"{len(df)} compounds "
        f"({int(df['is_binder'].sum())} binders, "
        f"{int((~df['is_binder']).sum())} suspected non-binders)."
    )


def main() -> None:
    args = parse_args()
    generate_virtual_screening_characterisation(
        benchmark_path=args.benchmark.resolve(),
        figures_dir=args.figures_dir.resolve(),
        tables_dir=args.tables_dir.resolve(),
    )


if __name__ == "__main__":
    main()

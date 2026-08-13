#!/usr/bin/env python3
"""Plot fragment-to-follow-on relationships for the enteroviral 2A benchmark.

The analysis combines the curated complex annotations with a structure-level
fragment similarity table produced during input preparation. The similarity
table must contain a complex identifier, the nearest curated fragment, and the
maximum ECFP4 Tanimoto similarity to that fragment.

Outputs
-------
Figures:
    fragment_series_distribution.png
    fragment_series_similarity.png
    fragment_series_examples.png

Tables:
    fragment_series_assignment_counts.csv
    fragment_series_similarity_summary.csv
    fragment_series_examples.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor, rdFMCS

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_ANNOTATED = (
    REPO_ROOT / "structure" / "processed_outputs" / "annotated_complexes.csv"
)
DEFAULT_SIMILARITY = (
    REPO_ROOT / "structure" / "processed_outputs" / "fragment_followon_similarity.csv"
)
DEFAULT_FIGURES_DIR = REPO_ROOT / "plotting" / "figures"
DEFAULT_TABLES_DIR = REPO_ROOT / "plotting" / "tables"

TOP_N_BARS = 9
TOP_N_SERIES = 3
N_EXAMPLES_PER_SERIES = 3
EXAMPLE_PERCENTILES = (0.75, 0.50, 0.35)
VIOLIN_ALPHA = 0.45
HIGHLIGHT_ALPHA = 0.25

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot fragment-to-follow-on relationships for the enteroviral 2A benchmark."
    )
    parser.add_argument(
        "--annotated",
        type=Path,
        default=DEFAULT_ANNOTATED,
        help="Curated annotated_complexes.csv.",
    )
    parser.add_argument(
        "--fragment-similarity",
        type=Path,
        default=DEFAULT_SIMILARITY,
        help="Fragment-to-follow-on similarity table produced during input preparation.",
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
        default=DEFAULT_TABLES_DIR,
        help="Output directory for CSV summary tables.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def parse_bool_column(series: pd.Series, name: str) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    parsed = values.map({"true": True, "false": False, "1": True, "0": False})
    invalid = parsed.isna()
    if invalid.any():
        examples = series.loc[invalid].astype(str).unique()[:10]
        raise ValueError(
            f"Missing or unrecognised boolean values in {name!r}: "
            + ", ".join(examples)
        )
    return parsed.astype(bool)


def canonical_smiles(smiles: object) -> str | None:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def pretty_id(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("A71EV2A-", "")


def mol_from_smiles(smiles: object) -> Chem.Mol | None:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    return mol


def _normalise_similarity(similarity: pd.DataFrame) -> pd.DataFrame:
    similarity = similarity.copy()

    if "complex_name" not in similarity.columns:
        if "complex_id" in similarity.columns:
            similarity = similarity.rename(columns={"complex_id": "complex_name"})
        else:
            raise ValueError(
                "Fragment similarity must contain 'complex_name' or 'complex_id'."
            )

    required = [
        "complex_name",
        "most_similar_fragment",
        "ECFP4_Tanimoto_Similarity",
    ]
    missing = set(required) - set(similarity.columns)
    if missing:
        raise ValueError(
            "Fragment similarity is missing columns: " + ", ".join(sorted(missing))
        )

    similarity = similarity[required].copy()
    similarity["complex_name"] = similarity["complex_name"].astype("string").str.strip()
    similarity["most_similar_fragment"] = (
        similarity["most_similar_fragment"].astype("string").str.strip()
    )
    similarity["ECFP4_Tanimoto_Similarity"] = pd.to_numeric(
        similarity["ECFP4_Tanimoto_Similarity"], errors="coerce"
    )

    duplicates = similarity["complex_name"].duplicated(keep=False)
    if duplicates.any():
        duplicate_ids = (
            similarity.loc[duplicates, "complex_name"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "Duplicate complex identifiers in fragment similarity: "
            + ", ".join(duplicate_ids[:20])
        )

    return similarity


def prepare_fragment_series_data(
    annotated_path: Path,
    similarity_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annotated = pd.read_csv(annotated_path)
    similarity = _normalise_similarity(pd.read_csv(similarity_path))

    required = {"complex_name", "smiles", "fragment_screen", "pb_valid", "artefact"}
    missing = required - set(annotated.columns)
    if missing:
        raise ValueError(
            "Annotated complexes are missing columns: " + ", ".join(sorted(missing))
        )

    annotated = annotated.copy()
    annotated["complex_name"] = annotated["complex_name"].astype("string").str.strip()
    annotated["fragment_screen"] = parse_bool_column(
        annotated["fragment_screen"], "fragment_screen"
    )
    annotated["pb_valid"] = parse_bool_column(annotated["pb_valid"], "pb_valid")
    annotated["artefact"] = parse_bool_column(annotated["artefact"], "artefact")
    annotated["filtered"] = ~annotated["pb_valid"] | annotated["artefact"]

    duplicates = annotated["complex_name"].duplicated(keep=False)
    if duplicates.any():
        duplicate_ids = (
            annotated.loc[duplicates, "complex_name"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "Duplicate complex_name values in annotated complexes: "
            + ", ".join(duplicate_ids[:20])
        )

    merged = annotated.merge(
        similarity,
        on="complex_name",
        how="left",
        validate="one_to_one",
    )

    curated = merged.loc[~merged["filtered"]].copy()
    fragments = curated.loc[curated["fragment_screen"]].copy()
    followons = curated.loc[~curated["fragment_screen"]].copy()

    missing_similarity = (
        followons[["most_similar_fragment", "ECFP4_Tanimoto_Similarity"]]
        .isna()
        .any(axis=1)
    )
    if missing_similarity.any():
        missing_ids = (
            followons.loc[missing_similarity, "complex_name"].astype(str).tolist()
        )
        raise ValueError(
            "Curated follow-on complexes are missing fragment similarity: "
            + ", ".join(missing_ids[:20])
        )

    invalid_similarity = ~followons["ECFP4_Tanimoto_Similarity"].between(0.0, 1.0)
    if invalid_similarity.any():
        invalid_ids = (
            followons.loc[invalid_similarity, "complex_name"].astype(str).tolist()
        )
        raise ValueError(
            "ECFP4 Tanimoto similarities outside [0, 1] for: "
            + ", ".join(invalid_ids[:20])
        )

    followons["compound_key"] = followons["smiles"].map(canonical_smiles)
    invalid_smiles = followons["compound_key"].isna()
    if invalid_smiles.any():
        invalid_ids = followons.loc[invalid_smiles, "complex_name"].astype(str).tolist()
        raise ValueError(
            "Could not canonicalise SMILES for follow-on complexes: "
            + ", ".join(invalid_ids[:20])
        )

    assignment_conflicts = (
        followons.groupby("compound_key")["most_similar_fragment"].nunique().gt(1)
    )
    if assignment_conflicts.any():
        raise ValueError(
            f"Found {int(assignment_conflicts.sum())} compounds with conflicting nearest-fragment assignments."
        )

    similarity_ranges = followons.groupby("compound_key")[
        "ECFP4_Tanimoto_Similarity"
    ].agg(lambda values: float(values.max() - values.min()))
    if (similarity_ranges > 1e-12).any():
        raise ValueError(
            "Repeated structures for the same compound have conflicting ECFP4 similarities."
        )

    unique_followons = (
        followons.sort_values(["compound_key", "complex_name"], kind="stable")
        .drop_duplicates("compound_key")
        .reset_index(drop=True)
    )

    fragment_lookup = (
        fragments[["complex_name", "smiles"]]
        .drop_duplicates("complex_name")
        .set_index("complex_name")["smiles"]
    )
    missing_fragments = sorted(
        set(unique_followons["most_similar_fragment"].dropna())
        - set(fragment_lookup.index)
    )
    if missing_fragments:
        raise ValueError(
            "Nearest-fragment assignments refer to non-curated or unknown fragments: "
            + ", ".join(map(str, missing_fragments[:20]))
        )

    return (
        curated,
        unique_followons,
        fragment_lookup.rename("fragment_smiles").reset_index(),
    )


def build_assignment_counts(unique_followons: pd.DataFrame) -> pd.DataFrame:
    counts = (
        unique_followons["most_similar_fragment"]
        .value_counts()
        .rename_axis("fragment_id")
        .reset_index(name="n_unique_compounds")
        .sort_values(
            ["n_unique_compounds", "fragment_id"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    total = int(counts["n_unique_compounds"].sum())
    counts["percent"] = 100.0 * counts["n_unique_compounds"] / total
    counts["cumulative_n"] = counts["n_unique_compounds"].cumsum()
    counts["cumulative_percent"] = 100.0 * counts["cumulative_n"] / total
    counts.insert(0, "rank", np.arange(1, len(counts) + 1))
    counts["fragment_id_short"] = counts["fragment_id"].map(pretty_id)
    return counts


def build_similarity_summary(unique_followons: pd.DataFrame) -> pd.DataFrame:
    grouped = unique_followons.groupby("most_similar_fragment")[
        "ECFP4_Tanimoto_Similarity"
    ]
    summary = grouped.agg(["count", "mean", "median", "min", "max"])
    summary["q1"] = grouped.quantile(0.25)
    summary["q3"] = grouped.quantile(0.75)
    summary["iqr"] = summary["q3"] - summary["q1"]
    summary = (
        summary.rename_axis("fragment_id")
        .reset_index()
        .sort_values(["count", "fragment_id"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary["fragment_id_short"] = summary["fragment_id"].map(pretty_id)
    return summary


def _top_with_other(counts: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if len(counts) <= top_n:
        return counts.copy()

    top = counts.head(top_n).copy()
    remaining = counts.iloc[top_n:]
    total = int(counts["n_unique_compounds"].sum())
    other_n = int(remaining["n_unique_compounds"].sum())
    other = pd.DataFrame(
        {
            "rank": [top_n + 1],
            "fragment_id": ["Other"],
            "n_unique_compounds": [other_n],
            "percent": [100.0 * other_n / total],
            "cumulative_n": [total],
            "cumulative_percent": [100.0],
            "fragment_id_short": ["Other"],
        }
    )
    return pd.concat([top, other], ignore_index=True)


def plot_fragment_series_distribution(
    counts: pd.DataFrame,
    save_path: Path,
) -> None:
    plot_df = _top_with_other(counts, TOP_N_BARS).iloc[::-1].reset_index(drop=True)
    values = plot_df["n_unique_compounds"].to_numpy()

    fig_height = max(4.2, 0.45 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(7.8, fig_height), dpi=300)

    y = np.arange(len(plot_df))
    bars = ax.barh(y, values, edgecolor="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["fragment_id_short"], fontsize=11)
    ax.set_xlabel("Unique follow-on compounds", fontsize=13)
    ax.set_ylabel("Nearest fragment", fontsize=13)

    x_max = max(values) if len(values) else 1
    for bar, value, percent in zip(bars, values, plot_df["percent"]):
        ax.text(
            bar.get_width() + 0.012 * x_max,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)} ({percent:.1f}%)",
            ha="left",
            va="center",
            fontsize=9.5,
        )

    ax.set_xlim(0, x_max * 1.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fragment_series_similarity(
    unique_followons: pd.DataFrame,
    counts: pd.DataFrame,
    fragment_lookup: pd.DataFrame,
    save_path: Path,
) -> None:
    top = counts.head(TOP_N_SERIES).merge(
        fragment_lookup,
        left_on="fragment_id",
        right_on="complex_name",
        how="left",
        validate="one_to_one",
    )

    plot_data: list[np.ndarray] = []
    labels: list[str] = []
    molecules: list[Chem.Mol | None] = []

    for row in top.itertuples(index=False):
        values = (
            unique_followons.loc[
                unique_followons["most_similar_fragment"].eq(row.fragment_id),
                "ECFP4_Tanimoto_Similarity",
            ]
            .astype(float)
            .to_numpy()
        )
        plot_data.append(values)
        labels.append(
            f"{pretty_id(row.fragment_id)}\n$n={len(values)}$ ({row.percent:.1f}%)"
        )
        molecules.append(mol_from_smiles(row.fragment_smiles))

    positions = np.arange(1, len(plot_data) + 1)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(plot_data)]
    rng = np.random.default_rng(3)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)
    parts = ax.violinplot(
        plot_data,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, colour in zip(parts["bodies"], colours):
        body.set_facecolor(colour)
        body.set_edgecolor("black")
        body.set_linewidth(0.7)
        body.set_alpha(VIOLIN_ALPHA)

    ax.boxplot(
        plot_data,
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

    for x, values in zip(positions, plot_data):
        sample = values
        if len(values) > 350:
            sample = rng.choice(values, size=350, replace=False)
        ax.scatter(
            rng.normal(x, 0.035, len(sample)),
            sample,
            s=6,
            alpha=0.20,
            c="black",
            linewidths=0,
            rasterized=True,
            zorder=3,
        )

    draw_options = Draw.MolDrawOptions()
    draw_options.padding = 0.03
    for x, mol in zip(positions, molecules):
        if mol is None:
            continue
        image = Draw.MolToImage(Chem.Mol(mol), size=(240, 150), options=draw_options)
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(np.asarray(image), zoom=0.42),
                (x, 0.84),
                xycoords="data",
                box_alignment=(0.5, 0.5),
                frameon=False,
                pad=0.0,
                annotation_clip=True,
                zorder=5,
            )
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Nearest curated fragment", fontsize=13)
    ax.set_ylabel("Maximum ECFP4 Tanimoto similarity", fontsize=13)
    ax.set_ylim(0.0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.15, top=0.97)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _bonds_for_atoms(mol: Chem.Mol, atom_ids: list[int]) -> list[int]:
    atoms = set(atom_ids)
    return [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms
    ]


def _align_and_highlight(
    seed_smiles: str, followon_smiles: str
) -> tuple[Chem.Mol, list[int], list[int]]:
    seed = mol_from_smiles(seed_smiles)
    followon = mol_from_smiles(followon_smiles)
    if seed is None or followon is None:
        raise ValueError("Could not parse a selected fragment-series SMILES.")

    pattern: Chem.Mol | None = None
    match = followon.GetSubstructMatch(seed)
    if match:
        pattern = seed
    else:
        try:
            mcs = rdFMCS.FindMCS(
                [seed, followon],
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
                timeout=10,
            )
            if mcs.smartsString and mcs.numAtoms >= 4:
                pattern = Chem.MolFromSmarts(mcs.smartsString)
                if pattern is not None:
                    match = followon.GetSubstructMatch(pattern)
        except Exception:
            pattern = None
            match = ()

    if pattern is not None and match:
        try:
            rdDepictor.GenerateDepictionMatching2DStructure(
                followon,
                seed,
                refPatt=pattern,
                acceptFailure=True,
            )
        except Exception:
            AllChem.Compute2DCoords(followon)

    atoms = list(match) if match else []
    return followon, atoms, _bonds_for_atoms(followon, atoms)


def select_fragment_series_examples(
    unique_followons: pd.DataFrame,
    counts: pd.DataFrame,
    fragment_lookup: pd.DataFrame,
) -> pd.DataFrame:
    fragment_smiles = fragment_lookup.set_index("complex_name")["fragment_smiles"]
    rows: list[dict[str, object]] = []

    for count_row in counts.head(TOP_N_SERIES).itertuples(index=False):
        fragment_id = count_row.fragment_id
        rows.append(
            {
                "series_fragment": fragment_id,
                "series_fragment_short": pretty_id(fragment_id),
                "role": "fragment",
                "complex_name": fragment_id,
                "smiles": fragment_smiles.loc[fragment_id],
                "tanimoto_to_fragment": np.nan,
                "series_n_unique_compounds": int(count_row.n_unique_compounds),
            }
        )

        series = (
            unique_followons.loc[
                unique_followons["most_similar_fragment"].eq(fragment_id)
            ]
            .sort_values(
                [
                    "ECFP4_Tanimoto_Similarity",
                    "compound_key",
                    "complex_name",
                ],
                ascending=[False, False, False],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        selected_indices: list[int] = []
        for percentile in EXAMPLE_PERCENTILES[:N_EXAMPLES_PER_SERIES]:
            index = int(round((1.0 - percentile) * (len(series) - 1)))
            if index not in selected_indices:
                selected_indices.append(index)

        for row in series.iloc[selected_indices].itertuples(index=False):
            rows.append(
                {
                    "series_fragment": fragment_id,
                    "series_fragment_short": pretty_id(fragment_id),
                    "role": "follow-on",
                    "complex_name": row.complex_name,
                    "smiles": row.smiles,
                    "tanimoto_to_fragment": float(row.ECFP4_Tanimoto_Similarity),
                    "series_n_unique_compounds": int(count_row.n_unique_compounds),
                }
            )

    return pd.DataFrame(rows)


def plot_fragment_series_examples(
    selected: pd.DataFrame,
    total_unique_followons: int,
    save_path: Path,
) -> None:
    series_ids = selected["series_fragment"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(series_ids),
        1,
        figsize=(12.5, 3.0 * len(series_ids)),
        dpi=300,
    )
    if len(series_ids) == 1:
        axes = [axes]

    highlight_colour = (0.9, 0.0, 0.0, HIGHLIGHT_ALPHA)

    for ax, fragment_id in zip(axes, series_ids):
        series = selected.loc[selected["series_fragment"].eq(fragment_id)].copy()
        fragment = series.loc[series["role"].eq("fragment")].iloc[0]
        fragment_mol = mol_from_smiles(fragment["smiles"])
        if fragment_mol is None:
            raise ValueError(f"Could not parse fragment SMILES for {fragment_id}.")

        molecules = [fragment_mol]
        legends = [f"{pretty_id(fragment_id)}\nfragment hit"]
        atom_lists: list[list[int]] = [[]]
        bond_lists: list[list[int]] = [[]]

        for row in series.loc[series["role"].eq("follow-on")].itertuples(index=False):
            mol, atoms, bonds = _align_and_highlight(fragment.smiles, row.smiles)
            molecules.append(mol)
            legends.append(
                f"{pretty_id(row.complex_name)}\nECFP4 similarity: {row.tanimoto_to_fragment:.2f}"
            )
            atom_lists.append(atoms)
            bond_lists.append(bonds)

        draw_options = Draw.MolDrawOptions()
        draw_options.legendFontSize = 42
        draw_options.legendFraction = 0.15
        draw_options.padding = 0.0

        image = Draw.MolsToGridImage(
            molecules,
            molsPerRow=len(molecules),
            subImgSize=(330, 250),
            legends=legends,
            highlightAtomLists=atom_lists,
            highlightBondLists=bond_lists,
            highlightAtomColors=[
                {atom: highlight_colour for atom in atoms} for atoms in atom_lists
            ],
            highlightBondColors=[
                {bond: highlight_colour for bond in bonds} for bonds in bond_lists
            ],
            useSVG=False,
            drawOptions=draw_options,
        )

        series_n = int(fragment["series_n_unique_compounds"])
        series_percent = 100.0 * series_n / total_unique_followons
        ax.imshow(np.asarray(image))
        ax.axis("off")
        ax.set_title(
            f"{pretty_id(fragment_id)} nearest-fragment group: "
            f"{series_n} ({series_percent:.1f}%) unique follow-on compounds",
            loc="left",
            fontsize=16,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_fragment_series_outputs(
    annotated_path: Path,
    similarity_path: Path,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    annotated_path = require_file(annotated_path, "Annotated complexes")
    similarity_path = require_file(similarity_path, "Fragment-follow-on similarity")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    _, unique_followons, fragment_lookup = prepare_fragment_series_data(
        annotated_path,
        similarity_path,
    )
    counts = build_assignment_counts(unique_followons)
    similarity_summary = build_similarity_summary(unique_followons).merge(
        fragment_lookup,
        left_on="fragment_id",
        right_on="complex_name",
        how="left",
        validate="one_to_one",
    )
    selected = select_fragment_series_examples(
        unique_followons,
        counts,
        fragment_lookup,
    )

    counts.merge(
        fragment_lookup,
        left_on="fragment_id",
        right_on="complex_name",
        how="left",
        validate="one_to_one",
    ).to_csv(tables_dir / "fragment_series_assignment_counts.csv", index=False)
    similarity_summary.to_csv(
        tables_dir / "fragment_series_similarity_summary.csv",
        index=False,
    )
    selected.to_csv(tables_dir / "fragment_series_examples.csv", index=False)

    plot_fragment_series_distribution(
        counts,
        figures_dir / "fragment_series_distribution.png",
    )
    plot_fragment_series_similarity(
        unique_followons,
        counts,
        fragment_lookup,
        figures_dir / "fragment_series_similarity.png",
    )
    plot_fragment_series_examples(
        selected,
        total_unique_followons=len(unique_followons),
        save_path=figures_dir / "fragment_series_examples.png",
    )


def main() -> None:
    args = parse_args()
    generate_fragment_series_outputs(
        annotated_path=args.annotated.resolve(),
        similarity_path=args.fragment_similarity.resolve(),
        figures_dir=args.figures_dir.resolve(),
        tables_dir=args.tables_dir.resolve(),
    )
    print("Generated fragment-series figures and tables.")


if __name__ == "__main__":
    main()

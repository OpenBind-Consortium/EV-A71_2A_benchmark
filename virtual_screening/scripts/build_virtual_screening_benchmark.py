#!/usr/bin/env python3
"""Build the OpenBind enteroviral 2A virtual-screening benchmark.

Inputs
------
1. SoakDB export containing the soaking-campaign experimental records.
2. Curated complex annotations containing crystallographic binding events.

The final benchmark is constructed at compound level using canonical isomeric
SMILES and contains:

    Name,SMILES,is_binder

Benchmark definition
--------------------
Binders
    - Start from all crystallographic hits in ``annotated_complexes.csv``.
    - Retain only PoseBusters-valid, non-artefactual binding events.
    - Exclude fragment-screen compounds.
    - Exclude small fragment-like compounds from the designed compound series
      when MW < 250 Da or HAC <= 18.
    - Deduplicate by canonical isomeric SMILES.

Suspected non-binders
    - Require a mounted crystal, successful diffraction, C121/C2 space group,
      resolution <= 2.75 A, successful DIMPLE processing, Dimple Rfree < 0.4,
      and RefinementOutcome == "7 - Analysed & Rejected".
    - Exclude controls and the CovHetLib library.
    - Remove ambiguous compound identifiers.
    - Exclude compounds matching any crystallographic hit by compound ID or
      canonical structure. All annotated hits are used for this exclusion,
      including hits that are later excluded from the curated binder set.
    - Exclude compounds with any reported LigandCC evidence anywhere in SoakDB,
      matched by compound ID or canonical structure.
    - Exclude compounds originating from designated fragment libraries.
    - Exclude small fragment-like compounds from the designed compound series
      when MW < 250 Da or HAC <= 18.
    - Deduplicate by canonical isomeric SMILES.

Combined benchmark
    - Require successful RDKit ETKDGv3 3D conformer generation.
    - Require unique compound identifiers and canonical structures.
"""

from __future__ import annotations

import argparse
import logging
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

LOGGER = logging.getLogger("openbind_vs_benchmark")
RDLogger.DisableLog("rdApp.warning")

MW_THRESHOLD = 250.0
HAC_THRESHOLD = 18
RESOLUTION_THRESHOLD = 2.75
RFREE_THRESHOLD = 0.4

FRAGMENT_LIBRARIES = {
    "DSIPoised",
    "Probing fragment all",
    "SpotXplorer",
}
DESIGNED_LIBRARIES = {
    "ASAPPTBOAM",
    "ASAPTBOAM",
    "ASAPPSOTNS",
}
EXCLUDED_LIBRARIES = {"CovHetLib"}
CONTROL_CODES = {"DMSO"}

REQUIRED_ANNOTATED_COLUMNS = {
    "complex_name",
    "smiles",
    "fragment_screen",
    "pb_valid",
    "artefact",
}
REQUIRED_SOAKDB_COLUMNS = {
    "ID",
    "LibraryName",
    "CompoundSMILES",
    "CompoundCode",
    "CrystalName",
    "MountingResult",
    "DataCollectionOutcome",
    "DataProcessingSpaceGroup",
    "DataProcessingResolutionHigh",
    "DataProcessingDimpleSuccessful",
    "DimpleRfree",
    "RefinementOutcome",
    "RefinementLigandCC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--soakdb", required=True, type=Path)
    parser.add_argument("--annotated-complexes", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("virtual_screening_benchmark.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("virtual_screening_benchmark_summary.csv"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260126,
        help="RDKit conformer-generation seed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, cpu_count() or 1),
        help="Parallel workers for 3D conformer checks.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def parse_bool(series: pd.Series, name: str) -> pd.Series:
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
            "y": True,
            "n": False,
            "t": True,
            "f": False,
        }
    ).astype("boolean")

    invalid = series.notna() & parsed.isna()
    if invalid.any():
        examples = series.loc[invalid].astype(str).unique()[:10]
        raise ValueError(
            f"Could not parse boolean values in {name!r}: " + ", ".join(examples)
        )

    return parsed


def canonical_smiles(smiles: object) -> str | None:
    if pd.isna(smiles):
        return None

    text = str(smiles).strip()
    if not text:
        return None

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def crystal_id_from_complex_name(name: object) -> str:
    """Map A71EV2A-x1234a-style complex IDs to SoakDB crystal IDs."""
    text = str(name).strip()
    return re.sub(r"([xX]\d+)[A-Za-z]$", r"\1", text)


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        if pd.notna(value):
            text = str(value).strip()
            if text:
                return text
    return ""


def molecular_properties(smiles: pd.Series) -> tuple[pd.Series, pd.Series]:
    mols = smiles.map(Chem.MolFromSmiles)

    if mols.isna().any():
        failed = smiles.loc[mols.isna()].astype(str).tolist()
        raise ValueError(
            "Could not calculate molecular properties for: " + ", ".join(failed[:10])
        )

    mw = mols.map(Descriptors.MolWt)
    hac = mols.map(lambda mol: mol.GetNumHeavyAtoms())
    return mw, hac


def is_fragment_like(mw: pd.Series, hac: pd.Series) -> pd.Series:
    return (mw < MW_THRESHOLD) | (hac <= HAC_THRESHOLD)


def map_crystals_to_compound_codes(soakdb: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for crystal_name, group in soakdb.groupby("CrystalName", dropna=True):
        crystal = str(crystal_name).strip()
        codes = sorted(
            {
                str(value).strip()
                for value in group["CompoundCode"]
                if pd.notna(value) and str(value).strip()
            }
        )

        if len(codes) > 1:
            raise ValueError(
                f"SoakDB crystal {crystal!r} maps to multiple compound codes: "
                + ", ".join(codes)
            )

        if codes:
            mapping[crystal] = codes[0]

    return mapping


def prepare_hit_evidence(
    annotated: pd.DataFrame,
    soakdb: pd.DataFrame,
    stats: dict[str, int],
) -> tuple[set[str], set[str]]:
    """Return structure and compound-ID evidence for any observed hit."""
    hits = annotated.copy()
    hits["canonical_smiles"] = hits["smiles"].map(canonical_smiles)

    if hits["canonical_smiles"].isna().any():
        names = hits.loc[hits["canonical_smiles"].isna(), "complex_name"]
        raise ValueError(
            "Invalid SMILES in annotated hit list: "
            + ", ".join(names.astype(str).tolist()[:10])
        )

    crystal_to_code = map_crystals_to_compound_codes(soakdb)
    hits["crystal_name"] = hits["complex_name"].map(crystal_id_from_complex_name)
    hits["compound_code"] = hits["crystal_name"].map(crystal_to_code)

    missing_codes = hits["compound_code"].isna()
    if missing_codes.any():
        crystals = hits.loc[missing_codes, "crystal_name"].drop_duplicates()
        raise ValueError(
            "Annotated hit crystals could not be mapped to SoakDB compound IDs: "
            + ", ".join(crystals.astype(str).tolist()[:10])
        )

    stats["annotated_hit_events"] = len(hits)
    stats["annotated_hit_unique_structures"] = hits["canonical_smiles"].nunique()
    stats["annotated_hit_unique_compound_ids"] = hits["compound_code"].nunique()

    return (
        set(hits["canonical_smiles"]),
        set(hits["compound_code"].astype(str)),
    )


def curate_binders(
    annotated: pd.DataFrame,
    stats: dict[str, int],
) -> pd.DataFrame:
    binders = annotated.copy()
    binders["pb_valid"] = parse_bool(binders["pb_valid"], "pb_valid")
    binders["artefact"] = parse_bool(binders["artefact"], "artefact")
    binders["fragment_screen"] = parse_bool(
        binders["fragment_screen"],
        "fragment_screen",
    )

    stats["binder_input_events"] = len(binders)

    binders = binders.loc[
        binders["pb_valid"].eq(True) & binders["artefact"].eq(False)
    ].copy()
    stats["binder_events_after_structural_curation"] = len(binders)

    binders["canonical_smiles"] = binders["smiles"].map(canonical_smiles)
    if binders["canonical_smiles"].isna().any():
        names = binders.loc[
            binders["canonical_smiles"].isna(),
            "complex_name",
        ]
        raise ValueError(
            "Invalid SMILES in curated binder set: "
            + ", ".join(names.astype(str).tolist()[:10])
        )

    fragment_conflicts = (
        binders.groupby("canonical_smiles")["fragment_screen"].nunique().gt(1)
    )
    if fragment_conflicts.any():
        raise ValueError(
            f"{int(fragment_conflicts.sum())} binder structures have conflicting "
            "fragment-screen annotations."
        )

    binders = (
        binders.sort_values(
            ["canonical_smiles", "complex_name"],
            kind="stable",
        )
        .drop_duplicates("canonical_smiles", keep="first")
        .copy()
    )
    stats["binder_unique_structures_after_deduplication"] = len(binders)

    fragment_screen = binders["fragment_screen"].astype(bool)
    stats["binder_fragment_screen_removed"] = int(fragment_screen.sum())
    binders = binders.loc[~fragment_screen].copy()

    binders["MW"], binders["HAC"] = molecular_properties(binders["canonical_smiles"])

    fragment_like = is_fragment_like(binders["MW"], binders["HAC"])
    stats["binder_fragment_like_removed"] = int(fragment_like.sum())
    binders = binders.loc[~fragment_like].copy()

    output = pd.DataFrame(
        {
            "Name": binders["complex_name"].astype("string").str.strip(),
            "SMILES": binders["canonical_smiles"],
            "is_binder": True,
        }
    )

    stats["binder_final_before_3d"] = len(output)
    return output


def prepare_ligandcc_evidence(
    soakdb: pd.DataFrame,
    stats: dict[str, int],
) -> tuple[set[str], set[str]]:
    values = soakdb["RefinementLigandCC"].astype("string").str.strip()
    has_ligandcc = values.notna() & values.ne("")

    evidence = soakdb.loc[has_ligandcc].copy()
    evidence["canonical_smiles"] = evidence["CompoundSMILES"].map(canonical_smiles)

    compound_codes = set(
        evidence["CompoundCode"]
        .dropna()
        .astype("string")
        .str.strip()
        .loc[lambda x: x.ne("")]
    )
    structures = set(evidence["canonical_smiles"].dropna())

    stats["soakdb_rows_with_ligandcc"] = int(has_ligandcc.sum())
    stats["soakdb_ligandcc_unique_structures"] = len(structures)
    stats["soakdb_ligandcc_unique_compound_ids"] = len(compound_codes)
    return structures, compound_codes


def filter_soakdb_experiments(
    soakdb: pd.DataFrame,
    stats: dict[str, int],
) -> pd.DataFrame:
    df = soakdb.copy()
    stats["soakdb_input_rows"] = len(df)

    mounted = (
        df["MountingResult"].astype("string").str.strip().str.startswith("OK", na=False)
    )
    df = df.loc[mounted].copy()
    stats["soakdb_rows_after_mounted"] = len(df)

    diffracted = (
        df["DataCollectionOutcome"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("success")
        .fillna(False)
    )
    df = df.loc[diffracted].copy()
    stats["soakdb_rows_after_diffraction"] = len(df)

    space_group = (
        df["DataProcessingSpaceGroup"]
        .astype("string")
        .str.upper()
        .str.replace(" ", "", regex=False)
    )
    correct_space_group = space_group.isin({"C121", "C2"})
    df = df.loc[correct_space_group].copy()
    stats["soakdb_rows_after_space_group"] = len(df)

    resolution = pd.to_numeric(
        df["DataProcessingResolutionHigh"],
        errors="coerce",
    )
    df = df.loc[resolution.le(RESOLUTION_THRESHOLD)].copy()
    stats["soakdb_rows_after_resolution"] = len(df)

    dimple_success = parse_bool(
        df["DataProcessingDimpleSuccessful"],
        "DataProcessingDimpleSuccessful",
    ).fillna(False)
    df = df.loc[dimple_success].copy()
    stats["soakdb_rows_after_dimple"] = len(df)

    rfree = pd.to_numeric(df["DimpleRfree"], errors="coerce")
    df = df.loc[rfree.lt(RFREE_THRESHOLD)].copy()
    stats["soakdb_rows_after_rfree"] = len(df)

    rejected = (
        df["RefinementOutcome"]
        .astype("string")
        .str.strip()
        .eq("7 - Analysed & Rejected")
        .fillna(False)
    )
    df = df.loc[rejected].copy()
    stats["soakdb_rows_after_rejected_status"] = len(df)

    return df


def curate_nonbinders(
    soakdb: pd.DataFrame,
    hit_smiles: set[str],
    hit_codes: set[str],
    ligandcc_smiles: set[str],
    ligandcc_codes: set[str],
    stats: dict[str, int],
) -> pd.DataFrame:
    negatives = filter_soakdb_experiments(soakdb, stats)
    negatives["source_row"] = negatives.index

    control_mask = negatives["LibraryName"].isna() | negatives["CompoundCode"].astype(
        "string"
    ).str.strip().isin(CONTROL_CODES).fillna(False)
    stats["nonbinder_control_rows_removed"] = int(control_mask.sum())
    negatives = negatives.loc[~control_mask].copy()

    library = negatives["LibraryName"].astype("string").str.strip()
    excluded_library = library.isin(EXCLUDED_LIBRARIES)
    stats["nonbinder_excluded_library_rows_removed"] = int(excluded_library.sum())
    negatives = negatives.loc[~excluded_library].copy()

    known_libraries = FRAGMENT_LIBRARIES | DESIGNED_LIBRARIES
    observed_libraries = set(negatives["LibraryName"].dropna().astype(str).str.strip())
    unknown_libraries = observed_libraries - known_libraries
    if unknown_libraries:
        raise ValueError(
            "Unclassified SoakDB libraries remain after exclusions: "
            + ", ".join(sorted(unknown_libraries))
        )

    crystal_values = negatives["CrystalName"]
    id_values = negatives["ID"]
    negatives["Name"] = [
        first_nonempty((code, crystal, row_id))
        for code, crystal, row_id in zip(
            negatives["CompoundCode"],
            crystal_values,
            id_values,
        )
    ]
    negatives["compound_code"] = (
        negatives["CompoundCode"].astype("string").str.strip().fillna("")
    )
    negatives["canonical_smiles"] = negatives["CompoundSMILES"].map(canonical_smiles)

    missing_name = negatives["Name"].eq("")
    invalid_smiles = negatives["canonical_smiles"].isna()
    stats["nonbinder_rows_removed_missing_name"] = int(missing_name.sum())
    stats["nonbinder_rows_removed_invalid_smiles"] = int(invalid_smiles.sum())
    negatives = negatives.loc[~missing_name & ~invalid_smiles].copy()

    structures_per_id = negatives.groupby("Name")["canonical_smiles"].nunique()
    ambiguous_ids = set(structures_per_id.loc[structures_per_id.gt(1)].index)
    ambiguous_mask = negatives["Name"].isin(ambiguous_ids)
    stats["nonbinder_ambiguous_compound_ids_removed"] = len(ambiguous_ids)
    stats["nonbinder_rows_removed_ambiguous_ids"] = int(ambiguous_mask.sum())
    negatives = negatives.loc[~ambiguous_mask].copy()

    hit_overlap = negatives["canonical_smiles"].isin(hit_smiles) | negatives[
        "compound_code"
    ].isin(hit_codes)
    stats["nonbinder_rows_removed_hit_overlap"] = int(hit_overlap.sum())
    stats["nonbinder_structures_removed_hit_overlap"] = int(
        negatives.loc[hit_overlap, "canonical_smiles"].nunique()
    )
    negatives = negatives.loc[~hit_overlap].copy()

    ligandcc_overlap = negatives["canonical_smiles"].isin(ligandcc_smiles) | negatives[
        "compound_code"
    ].isin(ligandcc_codes)
    stats["nonbinder_rows_removed_ligandcc"] = int(ligandcc_overlap.sum())
    stats["nonbinder_structures_removed_ligandcc"] = int(
        negatives.loc[ligandcc_overlap, "canonical_smiles"].nunique()
    )
    negatives = negatives.loc[~ligandcc_overlap].copy()

    stats["nonbinder_rows_before_structure_deduplication"] = len(negatives)

    # Record whether a structure occurred in a fragment library before
    # deduplicating, so structures appearing in both fragment and designed
    # libraries are still excluded.
    seen_in_fragment_library = (
        negatives.assign(
            from_fragment_library=negatives["LibraryName"].isin(FRAGMENT_LIBRARIES)
        )
        .groupby("canonical_smiles")["from_fragment_library"]
        .any()
    )

    negatives = (
        negatives.sort_values("source_row", kind="stable")
        .drop_duplicates("canonical_smiles", keep="first")
        .copy()
    )
    stats["nonbinder_unique_structures_after_deduplication"] = len(negatives)

    from_fragment_library = (
        negatives["canonical_smiles"]
        .map(seen_in_fragment_library)
        .fillna(False)
        .astype(bool)
    )
    stats["nonbinder_fragment_library_removed"] = int(from_fragment_library.sum())
    negatives = negatives.loc[~from_fragment_library].copy()

    negatives["MW"], negatives["HAC"] = molecular_properties(
        negatives["canonical_smiles"]
    )

    fragment_like = is_fragment_like(
        negatives["MW"],
        negatives["HAC"],
    )
    stats["nonbinder_fragment_like_removed"] = int(fragment_like.sum())
    negatives = negatives.loc[~fragment_like].copy()

    output = pd.DataFrame(
        {
            "Name": negatives["Name"].astype("string").str.strip(),
            "SMILES": negatives["canonical_smiles"],
            "is_binder": False,
        }
    )

    stats["nonbinder_final_before_3d"] = len(output)
    return output


def can_generate_3d(smiles: str, seed: int) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = int(seed)
    params.maxIterations = 200

    if hasattr(params, "timeout"):
        params.timeout = 5

    try:
        conformer_id = AllChem.EmbedMolecule(mol, params)
    except Exception:
        return False

    return conformer_id >= 0 and mol.GetNumConformers() > 0


def apply_3d_filter(
    benchmark: pd.DataFrame,
    seed: int,
    workers: int,
    stats: dict[str, int],
) -> pd.DataFrame:
    LOGGER.info(
        "Testing RDKit 3D conformer generation for %d compounds.",
        len(benchmark),
    )

    arguments = [(smiles, seed) for smiles in benchmark["SMILES"]]
    with Pool(processes=workers) as pool:
        results = pool.starmap(
            can_generate_3d,
            arguments,
            chunksize=32,
        )

    can_embed = pd.Series(results, index=benchmark.index, dtype=bool)
    failed = benchmark.loc[~can_embed]

    stats["rows_removed_failed_3d"] = len(failed)
    stats["binder_rows_removed_failed_3d"] = int(failed["is_binder"].sum())
    stats["nonbinder_rows_removed_failed_3d"] = int((~failed["is_binder"]).sum())

    if not failed.empty:
        LOGGER.warning(
            "Removed %d compounds that failed 3D generation: %s",
            len(failed),
            ", ".join(failed["Name"].astype(str).tolist()),
        )

    return benchmark.loc[can_embed].copy()


def validate_final_benchmark(df: pd.DataFrame) -> None:
    expected = ["Name", "SMILES", "is_binder"]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected benchmark columns: {list(df.columns)}")

    if df.isna().any().any():
        raise ValueError("Final benchmark contains missing values.")

    if df["Name"].astype(str).str.strip().eq("").any():
        raise ValueError("Final benchmark contains empty compound names.")

    if df["Name"].duplicated().any():
        names = df.loc[
            df["Name"].duplicated(keep=False),
            "Name",
        ].astype(str)
        raise ValueError(
            "Final benchmark contains duplicate compound names: "
            + ", ".join(names.unique()[:10])
        )

    if df["SMILES"].duplicated().any():
        smiles = df.loc[
            df["SMILES"].duplicated(keep=False),
            "SMILES",
        ].astype(str)
        raise ValueError(
            "Final benchmark contains duplicate canonical structures: "
            + ", ".join(smiles.unique()[:10])
        )

    binder_smiles = set(df.loc[df["is_binder"], "SMILES"])
    nonbinder_smiles = set(df.loc[~df["is_binder"], "SMILES"])
    overlap = binder_smiles & nonbinder_smiles
    if overlap:
        raise ValueError(
            "Binder/non-binder structural overlap remains in final benchmark."
        )


def build_benchmark(
    soakdb_path: Path,
    annotated_path: Path,
    output_path: Path,
    summary_path: Path,
    seed: int,
    workers: int,
) -> dict[str, int]:
    soakdb = pd.read_csv(soakdb_path, low_memory=False)
    annotated = pd.read_csv(annotated_path, low_memory=False)

    require_columns(
        soakdb,
        REQUIRED_SOAKDB_COLUMNS,
        "SoakDB",
    )
    require_columns(
        annotated,
        REQUIRED_ANNOTATED_COLUMNS,
        "Annotated complexes",
    )

    stats: dict[str, int] = {}

    LOGGER.info("Preparing crystallographic hit evidence.")
    hit_smiles, hit_codes = prepare_hit_evidence(
        annotated,
        soakdb,
        stats,
    )

    LOGGER.info("Curating crystallographic binders.")
    binders = curate_binders(annotated, stats)

    LOGGER.info("Preparing LigandCC evidence.")
    ligandcc_smiles, ligandcc_codes = prepare_ligandcc_evidence(
        soakdb,
        stats,
    )

    LOGGER.info("Curating suspected non-binders from SoakDB.")
    nonbinders = curate_nonbinders(
        soakdb,
        hit_smiles,
        hit_codes,
        ligandcc_smiles,
        ligandcc_codes,
        stats,
    )

    benchmark = pd.concat(
        [binders, nonbinders],
        ignore_index=True,
    )

    if benchmark["Name"].duplicated().any():
        names = benchmark.loc[
            benchmark["Name"].duplicated(keep=False),
            "Name",
        ].astype(str)
        raise ValueError(
            "Binder/non-binder assembly created duplicate compound names: "
            + ", ".join(names.unique()[:10])
        )

    if benchmark["SMILES"].duplicated().any():
        raise ValueError(
            "Binder/non-binder assembly contains duplicate structures. "
            "Hit exclusion should have removed all negative overlaps."
        )

    benchmark = apply_3d_filter(
        benchmark,
        seed,
        workers,
        stats,
    )

    benchmark = benchmark.sort_values(
        ["is_binder", "Name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    validate_final_benchmark(benchmark)

    stats["final_total"] = len(benchmark)
    stats["final_binders"] = int(benchmark["is_binder"].sum())
    stats["final_nonbinders"] = int((~benchmark["is_binder"]).sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark.to_csv(output_path, index=False)
    pd.DataFrame(
        {
            "metric": list(stats),
            "count": list(stats.values()),
        }
    ).to_csv(summary_path, index=False)

    return stats


def main() -> None:
    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    for path, label in [
        (args.soakdb, "SoakDB"),
        (args.annotated_complexes, "Annotated complexes"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    stats = build_benchmark(
        soakdb_path=args.soakdb,
        annotated_path=args.annotated_complexes,
        output_path=args.output,
        summary_path=args.summary,
        seed=args.seed,
        workers=args.workers,
    )

    print("\nVirtual-screening benchmark")
    print("-" * 42)
    for key in [
        "final_total",
        "final_binders",
        "final_nonbinders",
    ]:
        print(f"{key:34s} {stats[key]:6d}")

    print(f"\nBenchmark: {args.output}")
    print(f"Summary:   {args.summary}")


if __name__ == "__main__":
    main()

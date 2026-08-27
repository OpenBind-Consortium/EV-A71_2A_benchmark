#!/usr/bin/env python3
"""Build the curated compound-level affinity benchmark from structure-level inputs.

Inputs
------
1. A structure-level experimental reference containing:
   - fragalysis_code
   - smiles
   - experimental_pKD

2. Prediction CSV files containing either:
   - fragalysis_code
   - predicted_affinity

   or a wide docking table containing ``fragalysis_code`` and one or more
   recognised GNINA/Smina prediction columns.

3. ``annotated_complexes.csv``, used to remove exact binding events with
   ``pb_valid=False`` or ``artefact=True`` before compound-level aggregation.

Smina docking energies are converted from kcal/mol to pKD. Individual
prediction files are assumed to already contain predictions in their final
benchmark form.

Predictions from valid structures corresponding to the same compound are
averaged to produce one compound-level prediction per method.

Example
-------
python affinity/scripts/build_compound_level_affinity_predictions.py \
    affinity/predictions \
    --reference affinity/reference/fragalysis_compound_reference.csv \
    --annotated-complexes structure/processed_outputs/annotated_complexes.csv \
    --output affinity/outputs/compound_level_prediction_analysis.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

R_KCAL_MOL_K = 0.00198720425864083
DEFAULT_TEMPERATURE_K = 298.15

REFERENCE_COLUMNS = ("fragalysis_code", "smiles", "experimental_pKD")
INDIVIDUAL_COLUMNS = {"fragalysis_code", "predicted_affinity"}
ANNOTATION_COLUMNS = {"complex_name", "pb_valid", "artefact"}

# Accepted structure-level docking columns and their final method names.
METHOD_COLUMNS = {
    "gnina_crystal": "gnina_crystal",
    "gnina_crystal_minimized": "gnina_crystal_minimised",
    "gnina_crystal_minimised": "gnina_crystal_minimised",
    "gnina_redock": "gnina_redock",
    "gnina_fragment_crossdock": "gnina_fragment_crossdock",
    "smina_crystal": "smina_crystal",
    "smina_crystal_minimized": "smina_crystal_minimised",
    "smina_crystal_minimised": "smina_crystal_minimised",
    "smina_redock": "smina_redock",
    "smina_fragment_crossdock": "smina_fragment_crossdock",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the curated compound-level affinity benchmark."
    )
    parser.add_argument(
        "predictions_dir",
        type=Path,
        help="Directory containing structure-level prediction CSV files.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help=(
            "Structure-level affinity reference containing fragalysis_code, "
            "smiles, and experimental_pKD."
        ),
    )
    parser.add_argument(
        "--annotated-complexes",
        type=Path,
        required=True,
        help=(
            "Final annotated_complexes.csv. Exact complexes with pb_valid=False "
            "or artefact=True are removed before compound-level aggregation."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs/compound_level_prediction_analysis.csv"),
        help=(
            "Output CSV path (default: outputs/compound_level_prediction_analysis.csv)."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE_K,
        help=(
            "Temperature in kelvin for Smina conversion "
            f"(default: {DEFAULT_TEMPERATURE_K})."
        ),
    )
    return parser.parse_args()


def normalise_method_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def method_from_filename(path: Path) -> str:
    name = path.name
    if name.endswith("_predictions.csv"):
        name = name[: -len("_predictions.csv")]
    else:
        name = path.stem
    return normalise_method_name(name)


def smina_to_pkd(values: pd.Series, temperature: float) -> pd.Series:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero kelvin.")
    denominator = R_KCAL_MOL_K * temperature * math.log(10.0)
    return -pd.to_numeric(values, errors="coerce") / denominator


def join_codes(values: pd.Series) -> str:
    return "; ".join(sorted(set(values.dropna().astype(str))))


def read_reference(path: Path) -> pd.DataFrame:
    """Read and validate the structure-level experimental reference."""
    if not path.is_file():
        raise FileNotFoundError(f"Reference file not found: {path}")

    reference = pd.read_csv(path)
    missing_columns = set(REFERENCE_COLUMNS) - set(reference.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing_columns)}"
        )

    reference = reference[list(REFERENCE_COLUMNS)].copy()
    reference["fragalysis_code"] = (
        reference["fragalysis_code"].astype("string").str.strip()
    )
    reference["smiles"] = reference["smiles"].astype("string").str.strip()

    missing_code = reference["fragalysis_code"].isna() | reference[
        "fragalysis_code"
    ].eq("")
    if missing_code.any():
        raise ValueError(f"{path} contains missing or empty fragalysis_code values.")

    missing_smiles = reference["smiles"].isna() | reference["smiles"].eq("")
    if missing_smiles.any():
        raise ValueError(f"{path} contains missing or empty SMILES values.")

    raw_pkd = reference["experimental_pKD"].copy()
    reference["experimental_pKD"] = pd.to_numeric(raw_pkd, errors="coerce")

    invalid_pkd = raw_pkd.notna() & reference["experimental_pKD"].isna()
    if invalid_pkd.any():
        examples = reference.loc[invalid_pkd, "fragalysis_code"].astype(str).tolist()
        raise ValueError(
            "Non-numeric experimental_pKD values found for: " + ", ".join(examples[:20])
        )

    # Structures without a qualifying experimental affinity are not part of
    # the affinity benchmark reference.
    reference = reference.dropna(subset=["experimental_pKD"]).copy()

    conflicts = (
        reference.groupby("fragalysis_code")
        .agg(n_smiles=("smiles", "nunique"), n_pkd=("experimental_pKD", "nunique"))
        .query("n_smiles > 1 or n_pkd > 1")
    )
    if not conflicts.empty:
        raise ValueError(
            "Conflicting reference values for fragalysis_code(s): "
            + ", ".join(conflicts.index.astype(str)[:20])
        )

    reference = reference.drop_duplicates("fragalysis_code", keep="first")

    compound_conflicts = reference.groupby("smiles")["experimental_pKD"].nunique()
    compound_conflicts = compound_conflicts[compound_conflicts > 1]
    if not compound_conflicts.empty:
        raise ValueError(
            f"Found {len(compound_conflicts)} SMILES with conflicting "
            "experimental_pKD values."
        )

    return reference.reset_index(drop=True)


def parse_bool_column(series: pd.Series, column_name: str) -> pd.Series:
    """Parse a required annotation column as strict boolean values."""
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    parsed = normalized.map(mapping).astype("boolean")

    invalid = parsed.isna()
    if invalid.any():
        examples = series.loc[invalid].astype(str).unique()[:10]
        raise ValueError(
            f"Missing or unrecognised boolean values in {column_name!r}: "
            + ", ".join(examples)
        )

    return parsed


def filter_reference_by_annotations(
    reference: pd.DataFrame,
    annotations_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove structurally invalid binding events before aggregation."""
    if not annotations_path.is_file():
        raise FileNotFoundError(
            f"annotated_complexes.csv not found: {annotations_path}"
        )

    annotations = pd.read_csv(annotations_path)
    missing_columns = ANNOTATION_COLUMNS - set(annotations.columns)
    if missing_columns:
        raise ValueError(
            f"{annotations_path} is missing columns: {sorted(missing_columns)}"
        )

    annotations = annotations[["complex_name", "pb_valid", "artefact"]].copy()
    annotations["complex_name"] = (
        annotations["complex_name"].astype("string").str.strip()
    )

    missing_names = annotations["complex_name"].isna() | annotations["complex_name"].eq(
        ""
    )
    if missing_names.any():
        raise ValueError(
            f"{annotations_path} contains missing or empty complex_name values."
        )

    duplicates = annotations["complex_name"].duplicated(keep=False)
    if duplicates.any():
        duplicate_codes = (
            annotations.loc[duplicates, "complex_name"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "Duplicate complex_name values in annotated_complexes: "
            + ", ".join(duplicate_codes[:20])
        )

    annotations["pb_valid"] = parse_bool_column(annotations["pb_valid"], "pb_valid")
    annotations["artefact"] = parse_bool_column(annotations["artefact"], "artefact")

    reference = reference.copy()
    reference["fragalysis_code"] = (
        reference["fragalysis_code"].astype("string").str.strip()
    )

    n_structures_before = reference["fragalysis_code"].nunique()
    n_compounds_before = reference["smiles"].nunique()

    merged = reference.merge(
        annotations,
        left_on="fragalysis_code",
        right_on="complex_name",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_annotation = merged["_merge"].eq("left_only")
    if missing_annotation.any():
        missing_codes = (
            merged.loc[missing_annotation, "fragalysis_code"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        message = (
            "Reference Fragalysis code(s) missing from annotated_complexes: "
            + ", ".join(missing_codes[:20])
        )
        if len(missing_codes) > 20:
            message += f" ... ({len(missing_codes)} total)"
        raise ValueError(message)

    invalid = merged["pb_valid"].eq(False) | merged["artefact"].eq(True)
    valid_reference = (
        merged.loc[~invalid, reference.columns].copy().reset_index(drop=True)
    )

    report = {
        "n_structures_before": n_structures_before,
        "n_structures_after": valid_reference["fragalysis_code"].nunique(),
        "n_invalid_structures": int(invalid.sum()),
        "n_artefact": int(merged.loc[invalid, "artefact"].eq(True).sum()),
        "n_pb_invalid": int(merged.loc[invalid, "pb_valid"].eq(False).sum()),
        "n_both": int(
            (
                merged.loc[invalid, "artefact"].eq(True)
                & merged.loc[invalid, "pb_valid"].eq(False)
            ).sum()
        ),
        "n_compounds_before": n_compounds_before,
        "n_compounds_after": valid_reference["smiles"].nunique(),
    }
    report["n_compounds_removed"] = (
        report["n_compounds_before"] - report["n_compounds_after"]
    )

    return valid_reference, report


def validate_individual_method_column(
    frame: pd.DataFrame, path: Path, method: str
) -> None:
    """Check an optional method column against the filename-derived method name."""
    if "method" not in frame.columns:
        return

    values = frame["method"].dropna().astype(str).map(normalise_method_name).unique()
    if len(values) == 0:
        return
    if len(values) > 1:
        raise ValueError(f"{path.name} contains multiple values in its method column.")
    if values[0] != method:
        raise ValueError(
            f"Method mismatch in {path.name}: filename implies {method!r}, "
            f"but method column contains {values[0]!r}."
        )


def load_prediction_methods(
    csv_files: list[Path],
    temperature: float,
) -> dict[str, pd.DataFrame]:
    """Load supported individual and wide-format prediction files."""
    method_predictions: dict[str, pd.DataFrame] = {}

    for path in csv_files:
        frame = pd.read_csv(path)
        file_used = False

        if "fragalysis_code" in frame.columns:
            for input_column, method in METHOD_COLUMNS.items():
                if input_column not in frame.columns:
                    continue
                if method in method_predictions:
                    raise ValueError(
                        f"Method {method!r} appears more than once, including in "
                        f"{path.name}."
                    )

                values = frame[input_column]
                if method.startswith("smina_"):
                    values = smina_to_pkd(values, temperature)
                else:
                    values = pd.to_numeric(values, errors="coerce")

                if values.notna().sum() == 0:
                    raise ValueError(
                        f"No numeric predictions found in column {input_column!r} "
                        f"of {path.name}."
                    )

                method_predictions[method] = pd.DataFrame(
                    {
                        "fragalysis_code": frame["fragalysis_code"],
                        "predicted_affinity": values,
                    }
                )
                file_used = True

        if INDIVIDUAL_COLUMNS.issubset(frame.columns):
            method = method_from_filename(path)
            validate_individual_method_column(frame, path, method)

            if method in method_predictions:
                raise ValueError(
                    f"Method {method!r} appears more than once, including in {path.name}."
                )

            method_predictions[method] = frame[
                ["fragalysis_code", "predicted_affinity"]
            ].copy()
            file_used = True

        if not file_used:
            raise ValueError(
                f"Unsupported prediction file {path.name}. Expected either "
                "fragalysis_code + predicted_affinity, or fragalysis_code plus "
                "at least one recognised GNINA/Smina column."
            )

    if not method_predictions:
        raise ValueError("No prediction methods were found.")

    return method_predictions


def aggregate_method(
    reference: pd.DataFrame,
    method: str,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate one prediction method from structure level to compound level."""
    predictions = predictions[["fragalysis_code", "predicted_affinity"]].copy()
    predictions["fragalysis_code"] = (
        predictions["fragalysis_code"].astype("string").str.strip()
    )

    missing_codes = predictions["fragalysis_code"].isna() | predictions[
        "fragalysis_code"
    ].eq("")
    if missing_codes.any():
        raise ValueError(
            f"Missing or empty fragalysis_code values for method {method!r}."
        )

    predictions["predicted_affinity"] = pd.to_numeric(
        predictions["predicted_affinity"], errors="coerce"
    )
    if predictions["predicted_affinity"].notna().sum() == 0:
        raise ValueError(f"No numeric predictions found for method {method!r}.")

    duplicate_codes = predictions.loc[
        predictions["fragalysis_code"].duplicated(keep=False),
        "fragalysis_code",
    ].nunique()
    if duplicate_codes:
        print(
            f"Warning: {duplicate_codes} structures have duplicate rows for "
            f"{method}; predictions will be averaged."
        )

    # Average duplicate prediction rows for the same exact structure first.
    predictions = predictions.groupby("fragalysis_code", as_index=False).agg(
        predicted_affinity=("predicted_affinity", "mean")
    )

    # The reference has already been structurally QC-filtered, so predictions
    # for invalid complexes cannot contribute to the compound-level mean.
    merged = reference.merge(predictions, on="fragalysis_code", how="left")

    compound = (
        merged.groupby("smiles", as_index=False)
        .agg(
            experimental_pKD=("experimental_pKD", "first"),
            predicted_affinity=("predicted_affinity", "mean"),
            n_structures_with_reference=("fragalysis_code", "nunique"),
            n_structures_with_prediction=("predicted_affinity", "count"),
            fragalysis_codes=("fragalysis_code", join_codes),
        )
        .sort_values("smiles", kind="stable")
        .reset_index(drop=True)
    )
    compound.insert(0, "method", method)
    return compound


def validate_output(output: pd.DataFrame) -> None:
    """Validate the final compound-level table without hard-coding its size."""
    duplicates = output.duplicated(subset=["method", "smiles"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate method/compound rows found in final output.")

    if output["experimental_pKD"].isna().any():
        raise ValueError("Final output contains missing experimental_pKD values.")

    invalid_counts = (
        output["n_structures_with_prediction"] > output["n_structures_with_reference"]
    )
    if invalid_counts.any():
        raise ValueError(
            "Final output contains compounds with more prediction structures than "
            "reference structures."
        )


def print_annotation_report(report: dict[str, int]) -> None:
    print("\n=== Complex validity filtering ===")
    print(f"Reference structures before filtering: {report['n_structures_before']}")
    print(f"Valid reference structures retained: {report['n_structures_after']}")
    print(f"Invalid reference structures excluded: {report['n_invalid_structures']}")
    print(f"  artefact=True: {report['n_artefact']}")
    print(f"  pb_valid=False: {report['n_pb_invalid']}")
    print(f"  both: {report['n_both']}")
    print(f"Compounds before filtering: {report['n_compounds_before']}")
    print(f"Compounds after filtering: {report['n_compounds_after']}")
    print(f"Compounds removed: {report['n_compounds_removed']}")


def print_prediction_coverage(output: pd.DataFrame) -> None:
    print("\n=== Prediction coverage ===")
    for method, frame in output.groupby("method", sort=True):
        n_total = len(frame)
        n_predicted = int(frame["predicted_affinity"].notna().sum())
        print(f"{method}: {n_predicted}/{n_total} compounds")


def main() -> int:
    args = parse_args()

    predictions_dir = args.predictions_dir.resolve()
    if not predictions_dir.is_dir():
        raise NotADirectoryError(f"Predictions directory not found: {predictions_dir}")

    csv_files = sorted(predictions_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {predictions_dir}")

    reference = read_reference(args.reference.resolve())
    reference, annotation_report = filter_reference_by_annotations(
        reference,
        args.annotated_complexes.resolve(),
    )

    method_predictions = load_prediction_methods(csv_files, args.temperature)

    output = pd.concat(
        [
            aggregate_method(reference, method, predictions)
            for method, predictions in sorted(method_predictions.items())
        ],
        ignore_index=True,
    )
    validate_output(output)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print_annotation_report(annotation_report)
    print_prediction_coverage(output)

    print("\n=== Final dataset ===")
    print(f"Prediction files read: {len(csv_files)}")
    print(f"Structures used: {reference['fragalysis_code'].nunique()}")
    print(f"Compounds used: {reference['smiles'].nunique()}")
    print(f"Methods: {len(method_predictions)}")
    print(f"Output rows: {len(output)}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

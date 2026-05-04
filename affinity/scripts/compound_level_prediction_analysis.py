"""Create compound-level affinity prediction analysis table.

Expected directory layout:
affinity/
  reference/
    fragalysis_compound_reference.csv
  predictions/
    *_predictions.csv
  outputs/

Output is written to:
    outputs/compound_level_prediction_analysis.csv
"""

from pathlib import Path

import pandas as pd


def method_from_filename(path: Path) -> str:
    return path.name.removesuffix("_predictions.csv")


def join_codes(values: pd.Series) -> str:
    return "; ".join(sorted(set(values.dropna().astype(str))))


def main() -> None:
    affinity_dir = Path(__file__).resolve().parents[1]
    reference_path = affinity_dir / "reference" / "fragalysis_compound_reference.csv"
    predictions_dir = affinity_dir / "predictions"
    outputs_dir = affinity_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(reference_path)
    required_reference_cols = {"fragalysis_code", "smiles", "experimental_pKD"}
    missing_reference_cols = required_reference_cols - set(reference.columns)
    if missing_reference_cols:
        raise ValueError(
            f"{reference_path} is missing columns: {sorted(missing_reference_cols)}"
        )

    reference = reference.dropna(subset=["smiles", "experimental_pKD"]).copy()
    prediction_files = sorted(predictions_dir.glob("*_predictions.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No *_predictions.csv files found in {predictions_dir}")

    output_parts = []
    for prediction_path in prediction_files:
        method = method_from_filename(prediction_path)
        predictions = pd.read_csv(prediction_path)
        required_prediction_cols = {"fragalysis_code", "predicted_affinity"}
        missing_prediction_cols = required_prediction_cols - set(predictions.columns)
        if missing_prediction_cols:
            raise ValueError(
                f"{prediction_path} is missing columns: {sorted(missing_prediction_cols)}"
            )

        merged = reference.merge(
            predictions[["fragalysis_code", "predicted_affinity"]],
            on="fragalysis_code",
            how="left",
        )
        compound = (
            merged.groupby("smiles", as_index=False)
            .agg(
                experimental_pKD=("experimental_pKD", "first"),
                predicted_affinity=("predicted_affinity", "mean"),
                n_structures_with_reference=("fragalysis_code", "nunique"),
                n_structures_with_prediction=("predicted_affinity", "count"),
                fragalysis_codes=("fragalysis_code", join_codes),
            )
            .sort_values("smiles")
        )
        compound.insert(0, "method", method)
        output_parts.append(compound)

    output = pd.concat(output_parts, ignore_index=True)
    output.to_csv(outputs_dir / "compound_level_prediction_analysis.csv", index=False)


if __name__ == "__main__":
    main()

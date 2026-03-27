from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_feature_names(data_dir: Path) -> list[str]:
    features_csv = data_dir / "NUSW-NB15_features.csv"
    if not features_csv.exists():
        return []

    catalog = pd.read_csv(features_csv, encoding="latin1")
    if "Name" not in catalog.columns:
        return []

    return catalog["Name"].astype(str).str.strip().str.lower().tolist()


def _read_file(path: Path, names: list[str]) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    if names:
        try:
            df = pd.read_csv(
                path,
                header=0,
                names=names,
                encoding="latin1",
                encoding_errors="replace",
                low_memory=False,
            )
            if df.shape[1] == len(names):
                return df
        except Exception:
            pass

    try:
        return pd.read_csv(
            path,
            header=0,
            encoding="latin1",
            encoding_errors="replace",
            low_memory=False,
        )
    except Exception:
        return None


def load_dataset(data_dir: Path) -> pd.DataFrame:
    names = load_feature_names(data_dir)
    candidate_files = [
        data_dir / "UNSW-NB15_1.csv",
        data_dir / "UNSW-NB15_2.csv",
        data_dir / "UNSW-NB15_3.csv",
        data_dir / "UNSW-NB15_4.csv",
        data_dir / "UNSW_NB15_training-set.csv",
        data_dir / "UNSW_NB15_testing-set.csv",
    ]

    frames = []
    for path in candidate_files:
        df = _read_file(path, names)
        if df is not None:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No UNSW-NB15 CSV files found under {data_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = combined.columns.astype(str).str.strip().str.lower()

    if "label" not in combined.columns:
        raise ValueError("label column missing from loaded dataset")

    combined["label"] = pd.to_numeric(combined["label"], errors="coerce").fillna(0).astype(int)
    return combined


def build_attack_normal_subset(
    df: pd.DataFrame,
    normal_ratio: float = 0.5,
    normal_offset: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    attacks = df[df["label"] == 1]
    normals = df[df["label"] == 0]

    attack_count = len(attacks)
    target_from_attack_total = max(1, int(attack_count * normal_ratio) - normal_offset)

    target_from_max_attack_cat = None
    if "attack_cat" in df.columns:
        attack_cats = (
            df.loc[df["label"] == 1, "attack_cat"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        attack_cats = attack_cats.replace({"-": "unknown", "normal": "unknown"})
        cat_counts = attack_cats.value_counts()
        if not cat_counts.empty:
            target_from_max_attack_cat = max(1, int(cat_counts.max()) - normal_offset)

    # Prefer normal count tied to largest attack category to avoid over-dominant normal class.
    # Fall back to attack-total ratio rule when attack_cat is unavailable.
    if target_from_max_attack_cat is not None:
        target_normals = min(target_from_attack_total, target_from_max_attack_cat)
        print(
            f"Normal target computed from largest attack category and attack-total rule: "
            f"min({target_from_attack_total:,}, {target_from_max_attack_cat:,}) = {target_normals:,}"
        )
    else:
        target_normals = target_from_attack_total
        print(f"Normal target computed from attack-total rule: {target_normals:,}")

    if len(normals) <= target_normals:
        sampled_normals = normals
        print(
            f"Normals available {len(normals):,} < target {target_normals:,}; "
            "taking all normals."
        )
    else:
        sampled_normals = normals.sample(n=target_normals, random_state=random_state)
        print(
            f"Sampled {len(sampled_normals):,} normals out of {len(normals):,} "
            f"(target={target_normals:,}, attacks={attack_count:,})."
        )

    subset = pd.concat([attacks, sampled_normals], axis=0)
    subset = subset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return subset


def build_or_load_adjusted_subset(
    data_dir: Path,
    output_path: Path,
    normal_ratio: float = 0.5,
    normal_offset: int = 5000,
    random_state: int = 42,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    output_path = Path(output_path)

    if output_path.exists() and not force_rebuild:
        subset = pd.read_csv(output_path, low_memory=False)
        subset.columns = subset.columns.astype(str).str.strip().str.lower()
        if "label" not in subset.columns:
            raise ValueError(f"Saved subset at {output_path} does not contain a 'label' column")
        subset["label"] = pd.to_numeric(subset["label"], errors="coerce").fillna(0).astype(int)
        return subset

    full_df = load_dataset(Path(data_dir))
    subset = build_attack_normal_subset(
        full_df,
        normal_ratio=normal_ratio,
        normal_offset=normal_offset,
        random_state=random_state,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False)
    return subset


def main() -> None:
    data_dir = Path("data")
    output_path = Path("processed_data_full/attacks_normal_half.csv")

    subset = build_or_load_adjusted_subset(
        data_dir=data_dir,
        output_path=output_path,
        normal_ratio=0.5,
        normal_offset=5000,
        random_state=42,
        force_rebuild=True,
    )

    attack_count = int((subset["label"] == 1).sum())
    normal_count = int((subset["label"] == 0).sum())
    print(
        f"Saved adjusted subset to {output_path} with {len(subset):,} rows "
        f"(attacks={attack_count:,}, normals={normal_count:,})."
    )


if __name__ == "__main__":
    main()

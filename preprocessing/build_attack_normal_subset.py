from pathlib import Path

from preprocessing.label_adjustment import build_or_load_adjusted_subset


def main() -> None:
    output_path = Path("processed_data_full/attacks_normal_half.csv")
    subset = build_or_load_adjusted_subset(
        data_dir=Path("data"),
        output_path=output_path,
        normal_ratio=0.5,
        normal_offset=5000,
        random_state=42,
        force_rebuild=True,
    )

    attack_count = int((subset["label"] == 1).sum())
    normal_count = int((subset["label"] == 0).sum())
    print(
        f"Saved {output_path} with {len(subset):,} rows "
        f"(attacks={attack_count:,}, normals={normal_count:,})."
    )


if __name__ == "__main__":
    main()

from dataclasses import dataclass, field
from typing import List, Literal
from pathlib import Path

@dataclass
class PreprocessingConfig:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("processed_data"))
    use_label_adjusted_subset: bool = False
    adjusted_subset_filename: str = "attacks_normal_half.csv"
    adjusted_normal_ratio: float = 0.5
    adjusted_normal_offset: int = 5000
    adjusted_subset_force_rebuild: bool = False

    target_column: str = "label"
    attack_category_column: str = "attack_cat"

    # drop columns per analyzer + user rules
    columns_to_drop: List[str] = field(default_factory=lambda: [
        "ct_flw_http_mthd",  # drop candidate due to high missing values
        "is_ftp_login",      # drop candidate due to high missing values
        "dtcpb",             # low MI
        "stcpb",             # low MI
        "dloss",             # redundant with dpkts/dbytes
        "dpkts",             # redundant after packet_rate
        "dwin",              # remove one of the correlated pair
        "ltime",             # used for duration, then dropped
        "stime",             # drop after duration is created
        "dbytes",            # drop after byte_ratio is created
        "sbytes",            # drop after byte_ratio is created
    ])

    # threshold based decisions
    missing_drop_threshold: float = 0.2  # drop if >20 percent missing
    high_cardinality_threshold: int = 100  # target encode beyond this
    wide_range_threshold: float = 1e6  # user preference for wide-range scaling
    correlation_threshold: float = 0.90

    categorical_columns: List[str] = field(default_factory=lambda: [
        "ct_ftp_cmd"
    ])
    binary_columns: List[str] = field(default_factory=lambda: [
        "is_sm_ips_ports"
    ])

    encoding_strategy: Literal["label", "onehot", "target", "hybrid"] = "hybrid"
    target_encoding_columns: List[str] = field(default_factory=list)
    onehot_encoding_columns: List[str] = field(default_factory=lambda: ["ct_ftp_cmd"])

    scaling_strategy: Literal["standard", "minmax", "robust", "power", "quantile", "none"] = "robust"
    skip_scaling_columns: List[str] = field(default_factory=lambda: [
        "is_sm_ips_ports", "label"
    ])

    outlier_strategy: Literal["cap", "remove", "transform", "none"] = "cap"
    outlier_lower_percentile: float = 1.0
    outlier_upper_percentile: float = 99.0

    imbalance_strategy: Literal["smote", "adasyn", "undersample", "class_weight", "none"] = "smote"
    smote_sampling_strategy: str = "auto"
    smote_k_neighbors: int = 5
    smote_random_state: int = 42


    missing_strategy_numerical: Literal["median", "mean", "drop"] = "median"
    missing_strategy_categorical: Literal["mode", "drop"] = "mode"

    enable_feature_selection: bool = True
    mi_threshold: float = 0.005

    random_state: int = 42
    save_intermediate: bool = True

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        assert 0 < self.correlation_threshold <= 1.0
        assert 0 < self.outlier_lower_percentile < self.outlier_upper_percentile < 100
        assert 0 < self.adjusted_normal_ratio <= 1.0
        assert self.adjusted_normal_offset >= 0

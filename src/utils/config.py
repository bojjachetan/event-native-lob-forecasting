from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for src/utils/config.py. Install with `pip install pyyaml`."
    ) from exc


T = TypeVar("T")


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str
    description: str
    seed_list: List[int]
    output_dir: str
    device: str = "cpu"
    deterministic: bool = True


@dataclass
class RawPathsConfig:
    snapshot: str
    depth_buffer: str
    depth: str
    trades: str


@dataclass
class IntermediatePathsConfig:
    reconstructed_state: str
    events: str
    targets: str
    split_manifest: str


@dataclass
class LogsPathsConfig:
    eval_dir: str
    split_dir: str
    baseline_dir: str
    aligned_dir: str


@dataclass
class PathsConfig:
    raw: RawPathsConfig
    intermediate: IntermediatePathsConfig
    logs: LogsPathsConfig


@dataclass
class RuntimeConfig:
    python_bin: str = "python"
    num_workers: int = 4
    top_n_levels: int = 10
    max_levels_per_side: int = 5000
    strict_sequence: bool = True
    match_window_us: int = 250_000
    device: str = "cpu"
    max_ram_gb: float = 64.0
    dry_run_shapes: bool = False


@dataclass
class ReconstructionEventSchemaConfig:
    include_post_event_top10_state: bool = True
    columns: List[str] = field(default_factory=list)


@dataclass
class ReconstructionOrderBookConfig:
    top_n_levels: int = 10
    include_execution_sink: bool = True
    num_nodes: int = 21
    level_indexing: str = "one_based"


@dataclass
class ReconstructionTradeMatchingConfig:
    execution_inference: str = "passive_side_trade_alignment"
    match_window_us: int = 250_000


@dataclass
class ReconstructionMicrostructureFeaturesConfig:
    vpin_bucket_volume: float = 50.0
    vpin_window_size: int = 50
    levels: int = 10
    compute_microprice: bool = True
    compute_queue_imbalance: bool = True
    compute_mlofi: bool = True


@dataclass
class ReconstructionConfig:
    exchange: str
    symbol: str
    timestamp_unit: str
    event_schema: ReconstructionEventSchemaConfig
    order_book: ReconstructionOrderBookConfig
    trade_matching: ReconstructionTradeMatchingConfig
    microstructure_features: ReconstructionMicrostructureFeaturesConfig


@dataclass
class GraphConfig:
    num_nodes: int = 21
    num_levels: int = 10
    typed_edges: Dict[str, int] = field(default_factory=dict)


@dataclass
class FeaturesConfig:
    top_n_levels: int = 10
    event_feature_columns: List[str] = field(default_factory=list)
    graph: GraphConfig = field(default_factory=GraphConfig)


@dataclass
class RealizedVolatilityConfig:
    use_true_midprice: bool = True
    transform: str = "log"
    log_eps: float = 1.0e-8


@dataclass
class PriceMoveTargetConfig:
    enabled: bool = False
    label_column: str = "price_move_label"
    num_classes: int = 3


@dataclass
class TargetsConfig:
    horizons_seconds: List[int] = field(default_factory=lambda: [1, 5, 10])
    realized_volatility: RealizedVolatilityConfig = field(default_factory=RealizedVolatilityConfig)
    price_move: PriceMoveTargetConfig = field(default_factory=PriceMoveTargetConfig)


@dataclass
class ModelConfig:
    num_nodes: int = 21
    num_levels: int = 10
    num_event_types: int = 3
    numeric_msg_dim: int = 9
    memory_dim: int = 128
    time_dim: int = 64
    structure_embed_dim: int = 32
    structure_dim: int = 64
    raw_msg_dim: int = 64
    msg_hidden_dim: int = 128
    marked_hidden_dim: int = 256
    readout_dim: int = 256
    readout_heads: int = 4
    full_book_dropout: float = 0.10
    dropout: float = 0.10
    volatility_out_dim: int = 3
    price_move_out_dim: int = 3
    include_execution_sink: bool = True


@dataclass
class LossWeightsConfig:
    gap_nll: float = 1.0
    alpha_event_type: float = 1.0
    beta_location: float = 1.0
    gamma_volatility: float = 1.0
    delta_price_move: float = 1.0


@dataclass
class MemoryUpdateConfig:
    update_after_loss_only: bool = False
    reset_each_epoch: bool = True


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    val_fraction: float = 0.2
    patience: int = 2
    metric: str = "composite"
    min_delta: float = 1.0e-4
    lambda_rank: float = 0.25


@dataclass
class SupervisionConfig:
    replay_all_events: bool = True
    train_on_spine: bool = False
    eval_on_spine: bool = False
    mode: str = "all_events"
    interval_us: int = 250_000
    every_n: int = 10
    include_large_events: bool = True
    large_event_quantile: float = 0.95


@dataclass
class MonteCarloSamplesConfig:
    train: int = 5
    eval: int = 10
    final: int = 10


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 256
    chunk_size: int = 256
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    grad_clip_norm: float = 1.0
    monte_carlo_samples_k: int = 10
    optimizer: str = "AdamW"
    mixed_precision: bool = False
    truncated_bptt: bool = True
    normalize_vol_targets: bool = True
    target_scaler: str = "standard"
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    memory_update: MemoryUpdateConfig = field(default_factory=MemoryUpdateConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    supervision: SupervisionConfig = field(default_factory=SupervisionConfig)
    mc_samples: MonteCarloSamplesConfig = field(default_factory=MonteCarloSamplesConfig)


@dataclass
class SplitsConfig:
    scheme: str = "purged_walk_forward"
    embargo_us: int = 300_000_000
    train_window_us: int = 86_400_000_000
    test_window_us: int = 86_400_000_000
    step_us: int = 86_400_000_000
    anchored: bool = False
    min_train_events: int = 1_000
    min_test_events: int = 1_000


@dataclass
class MetricsConfig:
    event_modeling: List[str] = field(default_factory=list)
    downstream: List[str] = field(default_factory=list)
    optional_price_move: List[str] = field(default_factory=list)


@dataclass
class TrainingEvalConfig:
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    enable_price_move_head: bool = False
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


@dataclass
class BaselineAdapterConfig:
    deeplob_bucket_us: int = 100_000
    gcn_bucket_us: int = 1_000_000
    deeplob_seq_len: int = 50
    use_log_targets: bool = True


@dataclass
class BaselineTrainingConfig:
    batch_size: int = 64
    epochs: int = 30
    patience: int = 5
    val_fraction: float = 0.2
    min_val_samples: int = 32
    purge_us: int = 10_000_000
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    grad_clip_norm: float = 1.0


@dataclass
class BaselinesConfig:
    adapter: BaselineAdapterConfig = field(default_factory=BaselineAdapterConfig)
    training: BaselineTrainingConfig = field(default_factory=BaselineTrainingConfig)


@dataclass
class EvaluationConfig:
    rank_primary_metric: str = "spearman"
    report_rmse_mae: bool = True


@dataclass
class Config:
    experiment: ExperimentConfig
    paths: PathsConfig
    runtime: RuntimeConfig
    reconstruction: ReconstructionConfig
    features: FeaturesConfig
    targets: TargetsConfig
    model: ModelConfig
    training: TrainingConfig
    splits: SplitsConfig
    training_eval: TrainingEvalConfig
    baselines: BaselinesConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------


def _deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base



def _set_nested_key(data: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: MutableMapping[str, Any] = data
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], MutableMapping):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value



def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None

    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        pass

    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value



def _resolve_type(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        return args[0] if len(args) == 1 else Any
    return tp



def _convert_value(expected_type: Any, value: Any) -> Any:
    expected_type = _resolve_type(expected_type)
    origin = get_origin(expected_type)

    if is_dataclass(expected_type):
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected mapping for {expected_type}, got {type(value)}")
        return _from_dict(expected_type, value)

    if origin in {list, List, Sequence, tuple, Tuple}:
        (inner_type,) = get_args(expected_type)[:1] or (Any,)
        if not isinstance(value, list):
            raise TypeError(f"Expected list-like value, got {type(value)}")
        converted = [_convert_value(inner_type, item) for item in value]
        return converted if origin in {list, List, Sequence} else tuple(converted)

    if origin in {dict, Dict, Mapping}:
        args = get_args(expected_type)
        key_type = args[0] if len(args) > 0 else Any
        val_type = args[1] if len(args) > 1 else Any
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected mapping value, got {type(value)}")
        return {
            _convert_value(key_type, k) if key_type is not Any else k:
            _convert_value(val_type, v) if val_type is not Any else v
            for k, v in value.items()
        }

    if expected_type is Any:
        return value

    if expected_type in {int, float, str, bool}:
        return expected_type(value)

    return value



def _from_dict(cls: Type[T], data: Mapping[str, Any]) -> T:
    kwargs: Dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for field_name, field_def in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if field_name not in data:
            continue
        expected_type = type_hints.get(field_name, field_def.type)
        kwargs[field_name] = _convert_value(expected_type, data[field_name])
    return cls(**kwargs)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def load_yaml(path: Union[str, os.PathLike[str]]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Top-level YAML structure must be a mapping.")

    return data



def load_config(
    path: Union[str, os.PathLike[str]],
    overrides: Optional[Mapping[str, Any]] = None,
    cli_overrides: Optional[Sequence[str]] = None,
    as_dataclass_obj: bool = True,
) -> Union[Config, Dict[str, Any]]:
    raw = load_yaml(path)
    merged: Dict[str, Any] = copy.deepcopy(raw)

    if overrides:
        _deep_update(merged, overrides)

    if cli_overrides:
        for item in cli_overrides:
            if "=" not in item:
                raise ValueError(
                    f"Invalid override '{item}'. Expected dotted.path=value format."
                )
            key, value = item.split("=", 1)
            _set_nested_key(merged, key, _coerce_scalar(value))

    return _from_dict(Config, merged) if as_dataclass_obj else merged



def config_to_dict(config: Union[Config, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(config, dict):
        return config
    return config.to_dict()



def flatten_config(
    config: Union[Config, Dict[str, Any]],
    prefix: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    data = config_to_dict(config)
    flat: Dict[str, Any] = {}

    def _walk(cur: Any, cur_prefix: str) -> None:
        if isinstance(cur, Mapping):
            for k, v in cur.items():
                next_prefix = f"{cur_prefix}{sep}{k}" if cur_prefix else str(k)
                _walk(v, next_prefix)
        else:
            flat[cur_prefix] = cur

    _walk(data, prefix)
    return flat



def validate_config(config: Config) -> None:
    if config.experiment.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("experiment.device must be one of: auto, cpu, cuda, mps.")
    if config.runtime.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("runtime.device must be one of: auto, cpu, cuda, mps.")
    if config.splits.embargo_us < 300_000_000:
        raise ValueError("embargo_us must be at least 300,000,000 microseconds (5 minutes).")
    if config.training.monte_carlo_samples_k != 10:
        raise ValueError("This experiment manifest expects monte_carlo_samples_k = 10.")
    if config.training.mc_samples.final != 10:
        raise ValueError("Final paper mode expects training.mc_samples.final = 10.")
    if config.training.supervision.mode not in {
        "all_events",
        "every_n_events",
        "every_100ms",
        "every_250ms",
        "every_500ms",
        "last_event_per_bucket",
        "volatility_informative",
    }:
        raise ValueError("Unsupported training.supervision.mode.")
    if config.model.num_nodes != 2 * config.model.num_levels + 1:
        raise ValueError("model.num_nodes must equal 2 * model.num_levels + 1.")
    if config.runtime.top_n_levels != config.model.num_levels:
        raise ValueError("runtime.top_n_levels must match model.num_levels.")
    if len(config.experiment.seed_list) < 3 or len(config.experiment.seed_list) > 5:
        raise ValueError("experiment.seed_list must contain between 3 and 5 seeds.")
    if config.baselines.adapter.deeplob_bucket_us <= 0 or config.baselines.adapter.gcn_bucket_us <= 0:
        raise ValueError("Baseline bucket sizes must be positive.")
    if config.baselines.training.purge_us < max(config.targets.horizons_seconds) * 1_000_000:
        raise ValueError("baselines.training.purge_us must cover the maximum forward target horizon.")



def dump_json(config: Union[Config, Dict[str, Any]], indent: int = 2) -> str:
    return json.dumps(config_to_dict(config), indent=indent)



def dump_env(config: Union[Config, Dict[str, Any]], prefix: str = "CFG") -> str:
    flat = flatten_config(config)
    lines = []
    for key in sorted(flat.keys()):
        env_key = f"{prefix}_{key.replace('.', '_').upper()}"
        value = flat[key]
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        lines.append(f"{env_key}={json.dumps(value) if isinstance(value, str) else value}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and validate a YAML experiment manifest.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted override, e.g. training.epochs=10",
    )
    parser.add_argument(
        "--format",
        choices=["json", "env"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation checks",
    )
    args = parser.parse_args()

    config = load_config(args.config, cli_overrides=args.override, as_dataclass_obj=True)
    if not args.no_validate:
        validate_config(config)

    if args.format == "json":
        print(dump_json(config))
    else:
        print(dump_env(config))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Prepare, train, and select single-task ACT policies on the ActiveLab v3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import yaml

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, OBS_STATE

DATA_ROOT = Path("/home/yuzhaoshu/data/active_data/ActiveLab_Data")
MODEL_ROOT = Path("/home/yuzhaoshu/data/model/act")
TASKS = (
    "insert_the_funnel_into_the_measuring_cylinder",
    "insert_the_glass_rod_into_the_test_tube",
    "insert_the_titration_tube_into_the_conical_bottle",
)
CAMERA_KEYS = (
    "observation.images.head.active_view",
    "observation.images.left.wrist",
    "observation.images.right.wrist",
)
FEATURE_KEYS = (OBS_STATE, *CAMERA_KEYS, ACTION)
BAD_EPISODES = {
    "insert_the_funnel_into_the_measuring_cylinder": [5],
    "insert_the_glass_rod_into_the_test_tube": [20, 22, 23, 24],
    "insert_the_titration_tube_into_the_conical_bottle": [8, 34],
}
EXPECTED_STATE_NAMES = [
    *(f"left.joint{i}.angle_rad" for i in range(1, 7)),
    "left.gripper.pos",
    *(f"right.joint{i}.angle_rad" for i in range(1, 7)),
    "right.gripper.pos",
    *(f"head.joint{i}.angle_rad" for i in range(1, 7)),
]
FIXED_MODEL = {
    "state_action_dim": 20,
    "camera_keys": list(CAMERA_KEYS),
    "image_size_hw": [480, 640],
    "n_obs_steps": 1,
    "chunk_size": 100,
    "n_action_steps": 100,
    "vision_backbone": "resnet18",
    "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
    "dim_model": 512,
    "dim_feedforward": 3200,
    "n_heads": 8,
    "n_encoder_layers": 4,
    "n_decoder_layers": 1,
    "use_vae": True,
    "latent_dim": 32,
    "n_vae_encoder_layers": 4,
    "kl_weight": 10.0,
    "dropout": 0.1,
    "optimizer_lr": 1e-5,
    "optimizer_lr_backbone": 1e-5,
    "optimizer_weight_decay": 1e-4,
    "grad_clip_norm": 10.0,
}
DEFAULT_TRAINING = {
    "steps": 100_000,
    "batch_size": 8,
    "log_freq": 200,
    "save_freq": 20_000,
    "eval_steps": 0,
    "num_workers": 4,
}


def _flatten_names(names: Any) -> list[str] | None:
    if names is None:
        return None
    if isinstance(names, dict):
        flattened: list[str] = []
        for value in names.values():
            flattened.extend(value if isinstance(value, list | tuple) else [value])
        return [str(value) for value in flattened]
    if isinstance(names, list | tuple | np.ndarray):
        return [str(value) for value in names]
    return [str(names)]


def _scalar(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list | tuple) and len(value) == 1:
        value = value[0]
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        value = value.item()
    return value


def _episode_value(metadata: dict[str, Any], key: str) -> Any:
    if key not in metadata:
        raise ValueError(f"Episode metadata is missing {key!r}.")
    return _scalar(metadata[key])


def _metadata(root: Path, task: str) -> LeRobotDatasetMetadata:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose one of: {', '.join(TASKS)}")
    dataset_root = (root / task).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    metadata = LeRobotDatasetMetadata(repo_id=f"activelab/{task}", root=dataset_root)
    if metadata.info.codebase_version != CODEBASE_VERSION:
        raise ValueError(
            f"{dataset_root} uses LeRobot {metadata._version!r}; this recipe requires {CODEBASE_VERSION!r}."
        )
    if metadata.fps != 15:
        raise ValueError(f"Expected 15 Hz data for {task}, got {metadata.fps} Hz.")
    return metadata


def _validate_feature_schema(metadata: LeRobotDatasetMetadata) -> None:
    for key in (OBS_STATE, ACTION):
        feature = metadata.features.get(key)
        if feature is None or tuple(feature["shape"]) != (20,):
            raise ValueError(f"Expected {key} with shape (20,), got {feature!r}.")
        names = _flatten_names(feature.get("names"))
        if names is not None and names != EXPECTED_STATE_NAMES:
            raise ValueError(f"Unexpected {key} joint order: {names}")
    for key in CAMERA_KEYS:
        feature = metadata.features.get(key)
        if feature is None:
            raise ValueError(f"Required camera feature {key!r} is absent.")
        if feature.get("dtype") != "video" or tuple(feature["shape"]) != (480, 640, 3):
            raise ValueError(f"Expected RGB video feature {key} with shape (480, 640, 3), got {feature!r}.")


def _dataset_fingerprint(root: Path, task: str, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256()
    meta_root = root / task / "meta"
    files = sorted([meta_root / "info.json", *meta_root.glob("episodes/**/*.parquet")])
    for path in files:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(meta_root)).encode())
        digest.update(path.read_bytes())
    recipe = {
        "task": task,
        "dataset_root": str((root / task).expanduser().resolve()),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "excluded_episodes": BAD_EPISODES[task],
        "known_anomaly_episodes_included": [],
        "feature_keys": FEATURE_KEYS,
        "version": CODEBASE_VERSION,
    }
    digest.update(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()[:16]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _default_user_config(
    task: str,
    data_root: Path,
    model_root: Path,
    asset_dir: Path | None,
    seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return {
        "task": task,
        "data_root": str(data_root.expanduser().resolve()),
        "model_root": str(model_root.expanduser().resolve()),
        "asset_dir": str(asset_dir.resolve()) if asset_dir else None,
        "run_name": f"{task}-seed{seed}-{stamp}",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "feature_keys": list(FEATURE_KEYS),
        "model": FIXED_MODEL.copy(),
        "training": DEFAULT_TRAINING.copy(),
        "wandb": {"project": "activelab-act", "entity": None, "mode": "online", "disable_artifact": False},
    }


def _validate_user_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {"task", "data_root", "model_root", "run_name", "seed", "validation_fraction"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing fields: {missing}")
    if config["task"] not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {config['task']!r}.")
    if config.get("feature_keys", list(FEATURE_KEYS)) != list(FEATURE_KEYS):
        raise ValueError(
            "feature_keys are fixed for this recipe and must match the three cameras, state, and action."
        )
    if config.get("model", FIXED_MODEL) != FIXED_MODEL:
        raise ValueError(
            "model contains a non-baseline value; ACT architecture and action horizon are fixed."
        )
    training = {**DEFAULT_TRAINING, **config.get("training", {})}
    if training["steps"] <= 0 or training["batch_size"] <= 0:
        raise ValueError("training.steps and training.batch_size must be positive.")
    if training["save_freq"] <= 0 or training["eval_steps"] < 0 or training["log_freq"] <= 0:
        raise ValueError("training.log_freq and save_freq must be positive; eval_steps must be non-negative.")
    if float(config["validation_fraction"]) == 0.0 and training["eval_steps"] != 0:
        raise ValueError("Set training.eval_steps to 0 when validation_fraction is 0 (all normal episodes train).")
    wandb_config = {
        "project": "activelab-act",
        "entity": None,
        "mode": "online",
        "disable_artifact": False,
        **config.get("wandb", {}),
    }
    if wandb_config["mode"] != "online":
        raise ValueError("ActiveLab ACT training requires W&B online mode.")
    if not isinstance(wandb_config["disable_artifact"], bool):
        raise ValueError("wandb.disable_artifact must be a boolean.")
    config = dict(config)
    config["training"] = training
    config["wandb"] = wandb_config
    config["feature_keys"] = list(FEATURE_KEYS)
    config["model"] = FIXED_MODEL.copy()
    config["data_root"] = str(Path(config["data_root"]).expanduser().resolve())
    config["model_root"] = str(Path(config["model_root"]).expanduser().resolve())
    if config.get("asset_dir"):
        config["asset_dir"] = str(Path(config["asset_dir"]).expanduser().resolve())
    return config


def _compute_training_stats(
    metadata: LeRobotDatasetMetadata, train_episodes: list[int]
) -> tuple[dict[str, Any], int]:
    sums = {OBS_STATE: np.zeros(20, dtype=np.float64), ACTION: np.zeros(20, dtype=np.float64)}
    sums_sq = {OBS_STATE: np.zeros(20, dtype=np.float64), ACTION: np.zeros(20, dtype=np.float64)}
    minimum = {OBS_STATE: np.full(20, np.inf), ACTION: np.full(20, np.inf)}
    maximum = {OBS_STATE: np.full(20, -np.inf), ACTION: np.full(20, -np.inf)}
    count = 0
    table_cache: dict[Path, Any] = {}
    for episode_index in train_episodes:
        path = metadata.root / metadata.get_data_file_path(episode_index)
        if path not in table_cache:
            table_cache[path] = pq.read_table(path, columns=["episode_index", OBS_STATE, ACTION])
        table = table_cache[path]
        episode_table = table.filter(pc.equal(table["episode_index"], episode_index))
        state = np.asarray(episode_table[OBS_STATE].to_pylist(), dtype=np.float64).reshape(-1, 20)
        action = np.asarray(episode_table[ACTION].to_pylist(), dtype=np.float64).reshape(-1, 20)
        expected_length = int(_episode_value(metadata.episodes[episode_index], "length"))
        if len(state) != expected_length or len(action) != expected_length:
            raise ValueError(
                f"Episode {episode_index} has {len(state)} state and {len(action)} action rows; "
                f"metadata declares {expected_length}."
            )
        for key, values in ((OBS_STATE, state), (ACTION, action)):
            if not np.isfinite(values).all():
                raise ValueError(f"Episode {episode_index} contains non-finite values in {key}.")
            sums[key] += values.sum(axis=0)
            sums_sq[key] += np.square(values).sum(axis=0)
            minimum[key] = np.minimum(minimum[key], values.min(axis=0))
            maximum[key] = np.maximum(maximum[key], values.max(axis=0))
        count += len(state)
    if count < 2:
        raise ValueError("At least two training frames are required for normalization statistics.")
    stats: dict[str, Any] = {}
    for key in (OBS_STATE, ACTION):
        mean = sums[key] / count
        variance = np.maximum(sums_sq[key] / count - np.square(mean), 0.0)
        stats[key] = {
            "min": minimum[key].tolist(),
            "max": maximum[key].tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [count],
        }
    return stats, count


def _prepare_task(
    task: str,
    data_root: Path,
    model_root: Path,
    seed: int = 42,
    validation_fraction: float = 0.0,
) -> tuple[Path, Path]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1). Use 0 to train on every episode.")
    dataset_root = (data_root / task).expanduser().resolve()
    metadata = _metadata(data_root, task)
    _validate_feature_schema(metadata)
    # Keep the seven confirmed video/state-duration anomalies out of training.
    excluded = sorted(BAD_EPISODES[task])
    if any(episode < 0 or episode >= metadata.total_episodes for episode in excluded):
        raise ValueError(f"Configured excluded episodes are outside the dataset range: {excluded}")
    all_episodes = sorted(set(range(metadata.total_episodes)) - set(excluded))
    empty_episodes = [
        episode for episode in all_episodes if int(_episode_value(metadata.episodes[episode], "length")) <= 0
    ]
    if empty_episodes:
        raise ValueError(f"Episodes with no state/action frames cannot be trained: {empty_episodes}")
    shuffled = all_episodes.copy()
    random.Random(seed).shuffle(shuffled)
    n_validation = math.ceil(len(shuffled) * validation_fraction) if validation_fraction > 0 else 0
    if n_validation >= len(shuffled):
        raise ValueError("Split fraction leaves no training episodes.")
    train_episodes = sorted(shuffled[:-n_validation]) if n_validation else sorted(shuffled)
    validation_episodes = sorted(shuffled[-n_validation:]) if n_validation else []
    stats, train_frame_count = _compute_training_stats(metadata, train_episodes)
    fingerprint = _dataset_fingerprint(data_root, task, seed, validation_fraction)
    asset_dir = model_root / "datasets" / task / fingerprint
    manifest = {
        "schema_version": 1,
        "task": task,
        "dataset_root": str(dataset_root),
        "codebase_version": CODEBASE_VERSION,
        "fps": metadata.fps,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "dataset_fingerprint": fingerprint,
        "feature_keys": list(FEATURE_KEYS),
        "train_episode_indices": train_episodes,
        "validation_episode_indices": validation_episodes,
        "excluded_episode_indices": excluded,
        "known_anomaly_episode_indices_included": [],
        "train_episode_count": len(train_episodes),
        "validation_episode_count": len(validation_episodes),
        "train_frame_count": sum(
            int(_episode_value(metadata.episodes[index], "length")) for index in train_episodes
        ),
        "validation_frame_count": sum(
            int(_episode_value(metadata.episodes[index], "length")) for index in validation_episodes
        ),
        "normalization_train_frame_count": train_frame_count,
    }
    _atomic_json(asset_dir / "split_manifest.json", manifest)
    _atomic_json(asset_dir / "normalization_stats.json", stats)
    user_config = _default_user_config(task, data_root, model_root, asset_dir, seed, validation_fraction)
    config_path = asset_dir / "activelab_act.yaml"
    config_path.write_text(yaml.safe_dump(user_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(
        f"Prepared {task}: train={len(train_episodes)} episodes/{manifest['train_frame_count']} frames, "
        f"validation={len(validation_episodes)} episodes/{manifest['validation_frame_count']} frames, "
        f"excluded={len(excluded)} known duration anomalies: {excluded}.\n"
        f"Assets: {asset_dir}\nConfig: {config_path}"
    )
    return asset_dir, config_path


def _find_asset_dir(model_root: Path, task: str) -> Path:
    candidates = [
        path
        for path in (model_root / "datasets" / task).glob("*")
        if (path / "split_manifest.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No prepared assets for {task}; run `activelab-act prepare --task {task}` first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return _validate_user_config(config)


def _build_native_config(config: dict[str, Any], run_dir: Path, assets_dir: Path) -> dict[str, Any]:
    task = config["task"]
    dataset_root = Path(config["data_root"]) / task
    training = config["training"]
    wandb_config = config["wandb"]
    entity = os.environ.get("WANDB_ENTITY") or wandb_config.get("entity")
    return {
        "dataset": {
            "repo_id": f"activelab/{task}",
            "root": str(dataset_root),
            "feature_keys": list(FEATURE_KEYS),
            "split_manifest_path": str(assets_dir / "split_manifest.json"),
            "normalization_stats_path": str(assets_dir / "normalization_stats.json"),
            "use_imagenet_stats": True,
            "eval_split": config["validation_fraction"],
            "video_backend": "pyav",
        },
        "policy": {
            "type": "act",
            "n_obs_steps": 1,
            "chunk_size": 100,
            "n_action_steps": 100,
            "vision_backbone": "resnet18",
            "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
            "dim_model": 512,
            "dim_feedforward": 3200,
            "n_heads": 8,
            "n_encoder_layers": 4,
            "n_decoder_layers": 1,
            "use_vae": True,
            "latent_dim": 32,
            "n_vae_encoder_layers": 4,
            "kl_weight": 10.0,
            "dropout": 0.1,
            "optimizer_lr": 1e-5,
            "optimizer_lr_backbone": 1e-5,
            "optimizer_weight_decay": 1e-4,
            "use_amp": False,
            "push_to_hub": False,
        },
        "output_dir": str(run_dir),
        "job_name": config["run_name"],
        "resume": False,
        "seed": int(config["seed"]),
        "num_workers": int(training["num_workers"]),
        "batch_size": int(training["batch_size"]),
        "steps": int(training["steps"]),
        "log_freq": int(training["log_freq"]),
        "eval_steps": int(training["eval_steps"]),
        "save_checkpoint": True,
        "save_freq": int(training["save_freq"]),
        "use_policy_training_preset": True,
        "save_checkpoint_to_hub": False,
        "ema": {"enable": False},
        "wandb": {
            "enable": True,
            "project": wandb_config["project"],
            "entity": entity,
            "mode": "online",
            "disable_artifact": wandb_config["disable_artifact"],
        },
    }


def _validate_native_config(native_config: dict[str, Any]) -> None:
    """Parse and validate the generated config through LeRobot's own dataclasses."""
    import draccus

    import lerobot.policies.act.configuration_act  # noqa: F401
    from lerobot.configs.train import TrainPipelineConfig

    with tempfile.TemporaryDirectory(prefix="activelab-act-config-") as temporary_dir:
        config_path = Path(temporary_dir) / "lerobot_train.yaml"
        config_path.write_text(yaml.safe_dump(native_config, sort_keys=False), encoding="utf-8")
        parsed_config = draccus.parse(config_class=TrainPipelineConfig, config_path=config_path, args=[])
        parsed_config.validate()


def _runtime_env(model_root: Path, run_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
    env.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
    env.setdefault("NO_PROXY", "localhost,127.0.0.1")
    torch_cache = model_root / "pretrained" / "torch"
    hf_cache = model_root / "cache" / "huggingface"
    wandb_dir = run_dir / "wandb"
    for path in (torch_cache, hf_cache, wandb_dir):
        path.mkdir(parents=True, exist_ok=True)
    env["TORCH_HOME"] = str(torch_cache)
    env["HF_HOME"] = str(hf_cache)
    env["WANDB_DIR"] = str(wandb_dir)
    os.environ.update(env)
    return env


def _finalize_run_assets(staging_dir: Path, run_dir: Path) -> None:
    """Move staged metadata into the native output dir after LeRobot creates it."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("assets", "wandb"):
        source = staging_dir / name
        destination = run_dir / name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    assets_dir = run_dir / "assets"
    native_config_path = assets_dir / "lerobot_train.yaml"
    if native_config_path.is_file():
        native_config = yaml.safe_load(native_config_path.read_text(encoding="utf-8"))
        native_config["dataset"]["split_manifest_path"] = str(assets_dir / "split_manifest.json")
        native_config["dataset"]["normalization_stats_path"] = str(assets_dir / "normalization_stats.json")
        native_config_path.write_text(yaml.safe_dump(native_config, sort_keys=False), encoding="utf-8")

    for train_config_path in run_dir.glob("checkpoints/*/pretrained_model/train_config.json"):
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
        train_config["dataset"]["split_manifest_path"] = str(assets_dir / "split_manifest.json")
        train_config["dataset"]["normalization_stats_path"] = str(assets_dir / "normalization_stats.json")
        train_config_path.write_text(
            json.dumps(train_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    shutil.rmtree(staging_dir, ignore_errors=True)


def _train_from_user_config(config_path: Path, skip_eval: bool = False) -> None:
    config = _read_config(config_path)
    asset_dir = (
        Path(config["asset_dir"])
        if config.get("asset_dir")
        else _find_asset_dir(Path(config["model_root"]), config["task"])
    )
    for required_asset in ("split_manifest.json", "normalization_stats.json"):
        if not (asset_dir / required_asset).is_file():
            raise FileNotFoundError(f"Missing {asset_dir / required_asset}; rerun `activelab-act prepare`.")
    manifest = json.loads((asset_dir / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task") != config["task"]:
        raise ValueError(f"Prepared assets belong to {manifest.get('task')!r}, not {config['task']!r}.")
    if (
        manifest.get("seed") != config["seed"]
        or manifest.get("validation_fraction") != config["validation_fraction"]
    ):
        raise ValueError(
            "Config seed/validation_fraction do not match the prepared split; rerun prepare or restore them."
        )
    run_dir = Path(config["model_root"]) / "runs" / config["task"] / config["run_name"]
    if run_dir.exists():
        raise FileExistsError(f"Run directory already contains files: {run_dir}; choose another run_name.")
    # LeRobot refuses to start a fresh run when output_dir already exists. Keep
    # assets and W&B files alongside the target temporarily, then move them in
    # after the native trainer creates output_dir and its checkpoints.
    staging_dir = run_dir.parent / f".{run_dir.name}.activelab-staging"
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}; inspect or remove it first.")
    assets_dir = staging_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(asset_dir / "split_manifest.json", assets_dir / "split_manifest.json")
    shutil.copy2(asset_dir / "normalization_stats.json", assets_dir / "normalization_stats.json")
    shutil.copy2(config_path, assets_dir / "activelab_act.yaml")
    native_config = _build_native_config(config, run_dir, assets_dir)
    native_config_path = assets_dir / "lerobot_train.yaml"
    native_config_path.write_text(yaml.safe_dump(native_config, sort_keys=False), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[3]
    (assets_dir / "source_commit.txt").write_text(
        subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        + "\n",
        encoding="utf-8",
    )
    env = _runtime_env(Path(config["model_root"]), staging_dir)
    try:
        subprocess.run(
            [sys.executable, "-m", "lerobot.scripts.lerobot_train", f"--config_path={native_config_path}"],
            check=True,
            env=env,
            cwd=repo_root,
        )
    finally:
        _finalize_run_assets(staging_dir, run_dir)
    if not skip_eval:
        if manifest["validation_episode_indices"]:
            _evaluate_run(run_dir, device="auto", max_samples=0)
        else:
            print("No held-out episodes; skipped validation evaluation and checkpoint selection.")


def _raw_episode_table(metadata: LeRobotDatasetMetadata, episode_index: int, cache: dict[Path, Any]) -> Any:
    path = metadata.root / metadata.get_data_file_path(episode_index)
    if path not in cache:
        cache[path] = pq.read_table(path, columns=["episode_index", "index", "timestamp", OBS_STATE, ACTION])
    table = cache[path]
    return table.filter(pc.equal(table["episode_index"], episode_index))


def _validate_raw_episodes(metadata: LeRobotDatasetMetadata, episodes: list[int]) -> int:
    cache: dict[Path, Any] = {}
    frame_total = 0
    for episode_index in episodes:
        episode = metadata.episodes[episode_index]
        expected_length = int(_episode_value(episode, "length"))
        start = int(_episode_value(episode, "dataset_from_index"))
        stop = int(_episode_value(episode, "dataset_to_index"))
        if stop - start != expected_length:
            raise ValueError(
                f"Episode {episode_index}: dataset range [{start}, {stop}) != length {expected_length}."
            )
        table = _raw_episode_table(metadata, episode_index, cache)
        if table.num_rows != expected_length:
            raise ValueError(
                f"Episode {episode_index}: parquet has {table.num_rows} rows, expected {expected_length}."
            )
        indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
        timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        states = np.asarray(table[OBS_STATE].to_pylist(), dtype=np.float32).reshape(-1, 20)
        actions = np.asarray(table[ACTION].to_pylist(), dtype=np.float32).reshape(-1, 20)
        if indices[0] != start or indices[-1] + 1 != stop or np.any(np.diff(indices) != 1):
            raise ValueError(f"Episode {episode_index}: parquet frame indices do not match episode metadata.")
        if (
            not np.isfinite(timestamps).all()
            or not np.isfinite(states).all()
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Episode {episode_index}: timestamps/state/action contain NaN or Inf.")
        expected_dt = 1.0 / metadata.fps
        if len(timestamps) > 1 and not np.allclose(np.diff(timestamps), expected_dt, atol=1e-4, rtol=0.0):
            raise ValueError(f"Episode {episode_index}: timestamp spacing differs from {metadata.fps} Hz.")
        frame_total += expected_length
    return frame_total


def _check_files_and_offsets(metadata: LeRobotDatasetMetadata, episodes: list[int]) -> dict[str, Any]:
    data_refs: dict[int, set[int]] = {}
    video_refs: dict[str, dict[int, set[int]]] = {key: {} for key in CAMERA_KEYS}
    nonzero_offsets = 0
    for episode_index in episodes:
        episode = metadata.episodes[episode_index]
        data_path = metadata.root / metadata.get_data_file_path(episode_index)
        if not data_path.is_file():
            raise FileNotFoundError(f"Episode {episode_index} data file is missing: {data_path}")
        data_chunk = int(_episode_value(episode, "data/chunk_index"))
        data_file = int(_episode_value(episode, "data/file_index"))
        data_refs.setdefault(data_chunk, set()).add(data_file)
        for key in CAMERA_KEYS:
            video_path = metadata.root / metadata.get_video_file_path(episode_index, key)
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Episode {episode_index} video file for {key} is missing: {video_path}"
                )
            chunk = int(_episode_value(episode, f"videos/{key}/chunk_index"))
            file_index = int(_episode_value(episode, f"videos/{key}/file_index"))
            video_refs[key].setdefault(chunk, set()).add(file_index)
            offset = float(_episode_value(episode, f"videos/{key}/from_timestamp"))
            end_timestamp = float(_episode_value(episode, f"videos/{key}/to_timestamp"))
            if not math.isfinite(offset) or not math.isfinite(end_timestamp) or end_timestamp < offset:
                raise ValueError(f"Episode {episode_index}: invalid video timestamp bounds for {key}.")
            nonzero_offsets += int(abs(offset) > 1e-9)
    gaps: dict[str, list[int]] = {}
    for chunk, indices in data_refs.items():
        gaps[f"data/chunk-{chunk:03d}"] = sorted(set(range(min(indices), max(indices) + 1)) - indices)
    for key, chunks in video_refs.items():
        for chunk, indices in chunks.items():
            gaps[f"{key}/chunk-{chunk:03d}"] = sorted(set(range(min(indices), max(indices) + 1)) - indices)
    gaps = {name: missing for name, missing in gaps.items() if missing}
    return {"nonzero_video_offsets": nonzero_offsets, "referenced_file_index_gaps": gaps}


def _preflight(task: str, data_root: Path, asset_dir: Path, decode_endpoints: bool = True) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import shift_timestamps

    metadata = _metadata(data_root, task)
    _validate_feature_schema(metadata)
    manifest = json.loads((asset_dir / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task") != task:
        raise ValueError(f"Manifest task {manifest.get('task')!r} does not match {task!r}.")
    train = manifest["train_episode_indices"]
    validation = manifest["validation_episode_indices"]
    excluded = manifest["excluded_episode_indices"]
    for name, indices in (
        ("train_episode_indices", train),
        ("validation_episode_indices", validation),
        ("excluded_episode_indices", excluded),
    ):
        if not isinstance(indices, list) or any(type(index) is not int for index in indices):
            raise ValueError(f"Manifest {name} must be a list of integer episode indices.")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Manifest {name} contains duplicate episode indices.")
    if set(train) & set(validation) or set(train) & set(excluded) or set(validation) & set(excluded):
        raise ValueError("Train, validation, and excluded episode IDs are not disjoint.")
    expected_excluded = sorted(BAD_EPISODES[task])
    expected_valid = set(range(metadata.total_episodes)) - set(expected_excluded)
    if (
        set(train) | set(validation) != expected_valid
        or excluded != expected_excluded
        or manifest.get("known_anomaly_episode_indices_included") != []
    ):
        raise ValueError("Manifest must train on every normal episode and exclude the known duration anomalies.")
    if train != sorted(train) or validation != sorted(validation):
        raise ValueError("Manifest episode indices must be stored in sorted order.")
    seed = manifest.get("seed")
    validation_fraction = manifest.get("validation_fraction")
    if (
        type(seed) is not int
        or not isinstance(validation_fraction, int | float)
        or not 0 <= validation_fraction < 1
    ):
        raise ValueError("Manifest must contain an integer seed and validation_fraction in [0, 1).")
    shuffled = sorted(expected_valid)
    random.Random(seed).shuffle(shuffled)
    validation_count = math.ceil(len(shuffled) * float(validation_fraction)) if validation_fraction > 0 else 0
    expected_train = sorted(shuffled[:-validation_count]) if validation_count else sorted(shuffled)
    expected_validation = sorted(shuffled[-validation_count:]) if validation_count else []
    if train != expected_train or validation != expected_validation:
        raise ValueError("Manifest does not match the deterministic Random(seed) episode split.")
    expected_train_frames = sum(int(_episode_value(metadata.episodes[index], "length")) for index in train)
    expected_validation_frames = sum(
        int(_episode_value(metadata.episodes[index], "length")) for index in validation
    )
    if manifest.get("train_episode_count") != len(train) or manifest.get("validation_episode_count") != len(
        validation
    ):
        raise ValueError("Manifest episode counts do not match its episode index lists.")
    if (
        manifest.get("train_frame_count") != expected_train_frames
        or manifest.get("validation_frame_count") != expected_validation_frames
    ):
        raise ValueError("Manifest frame counts do not match the raw episode metadata.")
    all_referenced = sorted(set(train) | set(validation) | set(excluded))
    indexing = _check_files_and_offsets(metadata, all_referenced)
    frame_count = _validate_raw_episodes(metadata, train + validation)
    stats = json.loads((asset_dir / "normalization_stats.json").read_text(encoding="utf-8"))
    expected_stats, train_frame_count = _compute_training_stats(metadata, train)
    if manifest.get("normalization_train_frame_count") != train_frame_count:
        raise ValueError("Manifest normalization frame count does not match its train episodes.")
    for key in (OBS_STATE, ACTION):
        if key not in stats:
            raise ValueError(f"Normalization stats are missing {key!r}.")
        if stats[key].get("count") != [train_frame_count]:
            raise ValueError(f"Normalization stats frame count for {key!r} is not train-only.")
        for name in ("min", "max", "mean", "std"):
            actual = np.asarray(stats[key].get(name), dtype=np.float64)
            expected = np.asarray(expected_stats[key][name], dtype=np.float64)
            if actual.shape != (20,) or not np.isfinite(actual).all():
                raise ValueError(f"Normalization {name} for {key!r} must be a finite 20-dimensional vector.")
            if not np.allclose(actual, expected, rtol=1e-10, atol=1e-10):
                raise ValueError(
                    f"Normalization {name} for {key!r} differs from raw training-episode statistics."
                )
    print(
        f"Raw data OK: {len(train)} train + {len(validation)} validation episodes, "
        f"{frame_count} frames; excluded known duration anomalies={excluded}."
    )
    synthetic_shift = shift_timestamps([1.0, 2.0], 2.5)
    if synthetic_shift != [3.5, 4.5]:
        raise AssertionError(f"LeRobot video offset shift failed: {synthetic_shift}")
    print(
        f"File-index gaps in referenced ranges: {indexing['referenced_file_index_gaps'] or 'none'}; "
        f"nonzero video offsets in this dataset: {indexing['nonzero_video_offsets']}; "
        "native nonzero-offset timestamp shift check passed."
    )
    if not decode_endpoints:
        return
    dataset = LeRobotDataset(
        repo_id=f"activelab/{task}",
        root=data_root / task,
        episodes=sorted(set(train) | set(validation)),
        delta_timestamps={ACTION: [index / metadata.fps for index in range(100)]},
        video_backend="pyav",
        return_uint8=True,
        feature_keys=list(FEATURE_KEYS),
    )
    if dataset.meta.camera_keys != list(CAMERA_KEYS):
        raise ValueError(
            f"Feature projection returned cameras {dataset.meta.camera_keys}, expected {list(CAMERA_KEYS)}."
        )
    index_map = dataset.reader.absolute_to_relative_idx
    if index_map is None:
        raise RuntimeError("Episode-filtered reader did not build its absolute-to-relative index map.")
    decoded_samples = 0
    for episode_index in sorted(set(train) | set(validation)):
        episode = metadata.episodes[episode_index]
        first_index = int(_episode_value(episode, "dataset_from_index"))
        last_index = int(_episode_value(episode, "dataset_to_index")) - 1
        first_padding = max(0, 100 - int(_episode_value(episode, "length")))
        for absolute_index, expected_padding in ((first_index, first_padding), (last_index, 99)):
            item = dataset[index_map[absolute_index]]
            if tuple(item[OBS_STATE].shape) != (20,):
                raise ValueError(f"Episode {episode_index}: observation.state is not shape (20,).")
            if tuple(item[ACTION].shape) != (100, 20):
                raise ValueError(f"Episode {episode_index}: action window is not shape (100, 20).")
            actual_padding = int(item["action_is_pad"].sum().item())
            if actual_padding != expected_padding:
                raise ValueError(
                    f"Episode {episode_index}: expected {expected_padding} padded action slots, got {actual_padding}."
                )
            for key in CAMERA_KEYS:
                image = item[key]
                if tuple(image.shape) != (3, 480, 640):
                    raise ValueError(
                        f"Episode {episode_index}: decoded {key} has shape {tuple(image.shape)}."
                    )
            decoded_samples += 1
    print(
        f"Native LeRobot reader OK: decoded {decoded_samples} endpoint samples; "
        "state=(20,), action=(100,20), 3 RGB cameras, end padding mask correct."
    )


def _load_normalization_stats(path: Path, camera_keys: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    from lerobot.utils.constants import IMAGENET_STATS as IMAGE_STATS

    stats = {
        key: {
            name: torch.as_tensor(value, dtype=torch.float32)
            for name, value in feature_stats.items()
            if name != "count"
        }
        for key, feature_stats in json.loads(path.read_text(encoding="utf-8")).items()
    }
    for key in camera_keys:
        stats[key] = {name: torch.tensor(value, dtype=torch.float32) for name, value in IMAGE_STATS.items()}
    return stats


def _evaluate_checkpoint(
    checkpoint_dir: Path,
    device_name: str,
    max_samples: int,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.utils.collate import lerobot_collate_fn

    train_config_path = checkpoint_dir / "pretrained_model" / "train_config.json"
    saved_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    data_cfg = saved_config["dataset"]
    manifest = json.loads(Path(data_cfg["split_manifest_path"]).read_text(encoding="utf-8"))
    metadata = LeRobotDatasetMetadata(repo_id=data_cfg["repo_id"], root=data_cfg["root"])
    device = torch.device(
        device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = LeRobotDataset(
        repo_id=data_cfg["repo_id"],
        root=data_cfg["root"],
        episodes=manifest["validation_episode_indices"],
        delta_timestamps={ACTION: [index / metadata.fps for index in range(100)]},
        video_backend=data_cfg.get("video_backend", "pyav"),
        return_uint8=True,
        feature_keys=list(FEATURE_KEYS),
    )
    stats_path = Path(data_cfg["normalization_stats_path"])
    stats = _load_normalization_stats(stats_path, dataset.meta.camera_keys)
    dataset.meta.stats = stats
    policy = ACTPolicy.from_pretrained(checkpoint_dir / "pretrained_model")
    policy.config.device = str(device)
    policy.to(device)
    policy.eval()
    preprocessor, _ = make_act_pre_post_processors(policy.config, stats)
    loader = DataLoader(
        dataset,
        batch_size=int(saved_config["batch_size"]),
        shuffle=False,
        num_workers=int(saved_config.get("num_workers", 0)),
        collate_fn=lerobot_collate_fn,
        persistent_workers=False,
    )
    action_stats = stats[ACTION]
    mean = action_stats["mean"].to(device)
    std = action_stats["std"].to(device)
    dimension_groups = {
        "overall": list(range(20)),
        "left_arm_joints": list(range(0, 6)),
        "left_gripper": [6],
        "right_arm_joints": list(range(7, 13)),
        "right_gripper": [13],
        "active_view_arm_joints": list(range(14, 20)),
    }
    sums = dict.fromkeys(dimension_groups, 0.0)
    counts = dict.fromkeys(dimension_groups, 0)
    normalized_l1_sum = 0.0
    normalized_l1_count = 0
    processed_samples = 0
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            targets = batch[ACTION].to(device=device, dtype=torch.float32)
            padding = batch["action_is_pad"].to(device=device, dtype=torch.bool)
            for key in dataset.meta.camera_keys:
                if batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].to(dtype=torch.float32) / 255.0
            model_batch = preprocessor(batch)
            predicted_normalized = policy.predict_action_chunk(model_batch)
            normalized_error = (predicted_normalized - model_batch[ACTION]).abs()
            valid_steps = (~padding).unsqueeze(-1)
            normalized_l1_sum += float((normalized_error * valid_steps).sum().item())
            normalized_l1_count += int(valid_steps.sum().item()) * normalized_error.shape[-1]
            predicted = predicted_normalized * std + mean
            absolute_error = (predicted - targets).abs()
            for name, dimensions in dimension_groups.items():
                group_error = absolute_error[..., dimensions] * valid_steps
                sums[name] += float(group_error.sum().item())
                counts[name] += int(valid_steps.sum().item()) * len(dimensions)
            processed_samples += targets.shape[0]
            if max_samples > 0 and processed_samples >= max_samples:
                break
    metrics = {name: sums[name] / max(counts[name], 1) for name in dimension_groups}
    return {
        "step": int(checkpoint_dir.name),
        "checkpoint": str(checkpoint_dir),
        "validation_samples": processed_samples,
        "normalized_l1": normalized_l1_sum / max(normalized_l1_count, 1),
        "l1": metrics,
    }


def _evaluate_run(run_dir: Path, device: str = "auto", max_samples: int = 0) -> Path:
    model_root = run_dir.parents[2]
    _runtime_env(model_root, run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_dirs = sorted(
        [
            path
            for path in checkpoints_dir.iterdir()
            if path.is_dir()
            and path.name != "last"
            and (path / "pretrained_model" / "model.safetensors").is_file()
        ],
        key=lambda path: int(path.name),
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No loadable ACT checkpoints found under {checkpoints_dir}.")
    last_config = json.loads((checkpoint_dirs[-1] / "pretrained_model" / "train_config.json").read_text())
    data_cfg = last_config["dataset"]
    manifest = json.loads(Path(data_cfg["split_manifest_path"]).read_text(encoding="utf-8"))
    if not manifest.get("validation_episode_indices"):
        raise ValueError("This run trained on every non-excluded episode and has no held-out data to evaluate/select against.")
    user_config = yaml.safe_load((run_dir / "assets" / "activelab_act.yaml").read_text(encoding="utf-8"))
    results = [_evaluate_checkpoint(checkpoint, device, max_samples) for checkpoint in checkpoint_dirs]
    best = min(results, key=lambda result: result["normalized_l1"])
    selection = {
        "task": user_config["task"],
        "selected_checkpoint": best["checkpoint"],
        "selected_step": best["step"],
        "selection_metric": "validation/normalized_l1",
        "results": results,
    }
    output_path = run_dir / "selection.json"
    _atomic_json(output_path, selection)
    print(
        f"Selected step {best['step']} by normalized validation L1={best['normalized_l1']:.6f}; "
        f"active-view arm L1={best['l1']['active_view_arm_joints']:.6f} in dataset units.\nSaved: {output_path}"
    )
    wandb_cfg = last_config.get("wandb", {})
    run_id = wandb_cfg.get("run_id")
    if wandb_cfg.get("enable") and run_id:
        import wandb

        run = wandb.init(
            project=wandb_cfg["project"],
            entity=os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity"),
            id=run_id,
            resume="allow",
            job_type="checkpoint_selection",
            mode="online",
            dir=str(run_dir / "wandb"),
        )
        run.config.update(
            {
                "activelab": {
                    "task": user_config["task"],
                    "dataset_fingerprint": manifest["dataset_fingerprint"],
                    "split_seed": manifest["seed"],
                    "validation_fraction": manifest["validation_fraction"],
                    "train_episode_indices": manifest["train_episode_indices"],
                    "validation_episode_indices": manifest["validation_episode_indices"],
                    "excluded_episode_indices": manifest["excluded_episode_indices"],
                    "train_frame_count": manifest["train_frame_count"],
                    "validation_frame_count": manifest["validation_frame_count"],
                    "normalization_train_frame_count": manifest["normalization_train_frame_count"],
                    "model": user_config["model"],
                    "training": user_config["training"],
                }
            },
            allow_val_change=True,
        )
        run.define_metric("validation/step")
        run.define_metric("validation/*", step_metric="validation/step")
        for result in results:
            payload = {f"validation/l1/{key}": value for key, value in result["l1"].items()}
            payload["validation/normalized_l1"] = result["normalized_l1"]
            payload["validation/step"] = result["step"]
            run.log(payload)
        run.summary["selection/best_step"] = best["step"]
        run.summary["selection/best_validation_normalized_l1"] = best["normalized_l1"]
        run.summary["selection/best_validation_raw_l1"] = best["l1"]["overall"]
        run.summary["selection/active_view_arm_l1"] = best["l1"]["active_view_arm_joints"]
        run.finish()
    return output_path


def _launch_resume(checkpoint: Path, skip_eval: bool = False) -> None:
    train_config_path = checkpoint
    if checkpoint.is_dir():
        if checkpoint.name == "pretrained_model":
            train_config_path = checkpoint / "train_config.json"
        else:
            train_config_path = checkpoint / "pretrained_model" / "train_config.json"
    if not train_config_path.is_file():
        raise FileNotFoundError(f"Cannot find native train_config.json at {train_config_path}.")
    repo_root = Path(__file__).resolve().parents[3]
    saved = json.loads(train_config_path.read_text(encoding="utf-8"))
    output_dir = Path(saved["output_dir"])
    model_root = output_dir.parents[2]
    env = _runtime_env(model_root, output_dir)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            "--resume=true",
            f"--config_path={train_config_path}",
        ],
        check=True,
        env=env,
        cwd=repo_root,
    )
    if not skip_eval:
        manifest_path = Path(saved["dataset"]["split_manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("validation_episode_indices"):
            _evaluate_run(output_dir)
        else:
            print("No held-out episodes; skipped validation evaluation and checkpoint selection.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="activelab-act", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="validate the v3 metadata and create split/stat assets")
    prepare.add_argument("--task", choices=TASKS, required=True)
    prepare.add_argument("--data-root", type=Path, default=DATA_ROOT)
    prepare.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--validation-fraction", type=float, default=0.0)
    preflight = commands.add_parser("preflight", help="validate parquet, indexes, videos, windows and masks")
    preflight.add_argument("--task", choices=TASKS, required=True)
    preflight.add_argument("--data-root", type=Path, default=DATA_ROOT)
    preflight.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    preflight.add_argument("--asset-dir", type=Path)
    preflight.add_argument("--no-decode-endpoints", action="store_true")
    check_config = commands.add_parser("config-check", help="parse and validate an ActiveLab ACT YAML")
    check_config.add_argument("--config", type=Path, required=True)
    train = commands.add_parser("train", help="launch native LeRobot ACT training from an ActiveLab YAML")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--skip-eval", action="store_true")
    train_all = commands.add_parser("train-all", help="prepare and train all three tasks sequentially")
    train_all.add_argument(
        "--config", type=Path, required=True, help="YAML template with the desired training settings"
    )
    train_all.add_argument("--skip-eval", action="store_true")
    resume = commands.add_parser(
        "resume", help="resume native optimizer/sampler/RNG/W&B state from a checkpoint"
    )
    resume.add_argument("--checkpoint", type=Path, required=True)
    resume.add_argument("--skip-eval", action="store_true", help="skip checkpoint selection after resuming")
    evaluate = commands.add_parser(
        "evaluate", help="evaluate saved checkpoints and select the best validation L1"
    )
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate.add_argument("--max-samples", type=int, default=0, help="0 evaluates every validation frame")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        _prepare_task(args.task, args.data_root, args.model_root, args.seed, args.validation_fraction)
    elif args.command == "preflight":
        asset_dir = args.asset_dir or _find_asset_dir(args.model_root, args.task)
        _preflight(args.task, args.data_root, asset_dir, decode_endpoints=not args.no_decode_endpoints)
    elif args.command == "config-check":
        config = _read_config(args.config.expanduser().resolve())
        asset_dir = (
            Path(config["asset_dir"])
            if config.get("asset_dir")
            else _find_asset_dir(Path(config["model_root"]), config["task"])
        )
        for asset_name in ("split_manifest.json", "normalization_stats.json"):
            if not (asset_dir / asset_name).is_file():
                raise FileNotFoundError(
                    f"Missing {asset_dir / asset_name}; run `activelab-act prepare` first."
                )
        native = _build_native_config(
            config,
            Path(config["model_root"]) / "runs" / config["task"] / config["run_name"],
            asset_dir,
        )
        _validate_native_config(native)
        print(f"Config OK for {config['task']} using assets {asset_dir}.")
        print(yaml.safe_dump(native, sort_keys=False))
    elif args.command == "train":
        _train_from_user_config(args.config.expanduser().resolve(), args.skip_eval)
    elif args.command == "train-all":
        template = _read_config(args.config.expanduser().resolve())
        for task in TASKS:
            asset_dir, config_path = _prepare_task(
                task,
                Path(template["data_root"]),
                Path(template["model_root"]),
                int(template["seed"]),
                float(template["validation_fraction"]),
            )
            config = dict(template)
            config["task"] = task
            config["asset_dir"] = str(asset_dir)
            config["run_name"] = (
                f"{task}-seed{template['seed']}-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}"
            )
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            _train_from_user_config(config_path, args.skip_eval)
    elif args.command == "resume":
        _launch_resume(args.checkpoint.expanduser().resolve(), args.skip_eval)
    elif args.command == "evaluate":
        _evaluate_run(args.run_dir.expanduser().resolve(), args.device, args.max_samples)
    else:
        raise AssertionError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    main()

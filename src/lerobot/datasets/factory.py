#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import math
from pathlib import Path
from pprint import pformat
from typing import Literal, cast

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import (
    ACTION,
    DEFAULT_FEATURES,
    IMAGENET_STATS,
    OBS_IMAGE,
    OBS_PREFIX,
    OBS_STATE,
    REWARD,
)

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .storage import DEFAULT_STORAGE_FORMAT, is_bucket_root, load_dataset_metadata
from .streaming_dataset import DEFAULT_STREAMING_SEED, StreamingLeRobotDataset
from .utils import resolve_episode_indices

logger = logging.getLogger(__name__)


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig,
    ds_meta: LeRobotDatasetMetadata,
    rename_map: dict[str, str] | None = None,
) -> dict[str, list[float]] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list[float]] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    # Only policies that opt into modality-specific history (currently Pi05 with MEM)
    # define these; everything else falls back to the shared observation indices.
    explicit_image_indices = getattr(cfg, "image_observation_delta_indices", None)
    image_indices = (
        explicit_image_indices if explicit_image_indices is not None else cfg.observation_delta_indices
    )
    explicit_state_indices = getattr(cfg, "state_observation_delta_indices", None)
    state_indices = (
        explicit_state_indices if explicit_state_indices is not None else cfg.observation_delta_indices
    )

    delta_timestamps: dict[str, list[float]] = {}
    matched_image_keys: list[str] = []
    for key in ds_meta.features:
        policy_key = (rename_map or {}).get(key, key)
        if policy_key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if policy_key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        # `OBS_IMAGE` matches both the `observation.image` and `observation.images.<cam>`
        # conventions; matching `OBS_IMAGES` alone would silently give singular-key
        # datasets no image history at all.
        if policy_key.startswith(OBS_IMAGE):
            indices = image_indices
            matched_image_keys.append(key)
        elif policy_key == OBS_STATE:
            indices = state_indices
        else:
            indices = cfg.observation_delta_indices if policy_key.startswith(OBS_PREFIX) else None
        if indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in indices]

    # A policy asking for an image history that no dataset key can supply would train
    # on single frames without any error, so fail instead of degrading silently.
    if explicit_image_indices is not None and len(explicit_image_indices) > 1 and not matched_image_keys:
        raise ValueError(
            f"{type(cfg).__name__} requests {len(explicit_image_indices)} history frames per camera, but no "
            f"dataset feature maps to an image key. Dataset features: {sorted(ds_meta.features)}. "
            "Image keys must be named `observation.image*` after applying `--rename_map`."
        )

    if len(delta_timestamps) == 0:
        return None

    return delta_timestamps


def make_dataset(
    cfg: TrainPipelineConfig, episodes: list[int] | None = None
) -> LeRobotDataset | StreamingLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | StreamingLeRobotDataset: a StreamingLeRobotDataset when `cfg.dataset.streaming` is set.
    """
    # On the "gpu" backend the workers only decode; lerobot-train augments each batch on the policy device.
    image_transforms_cfg = cfg.dataset.image_transforms
    image_transforms = (
        ImageTransforms(image_transforms_cfg)
        if image_transforms_cfg.enable and image_transforms_cfg.backend == "dataloader"
        else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        root, repo_type_name = cfg.dataset.root, cfg.dataset.repo_type
        if cfg.dataset.streaming and is_bucket_root(cfg.dataset.repo_id, root):
            # Streaming reads default-format buckets directly, so a bucket URI root selects
            # bucket mode instead of the object-store localization used by other formats.
            root, repo_type_name = None, "bucket"
        repo_type = cast(Literal["dataset", "bucket"], repo_type_name)
        # Storage-aware loader: same as LeRobotDatasetMetadata(...), plus support
        # for datasets whose root is an object-store URI (e.g. ``hf://``).
        ds_meta = load_dataset_metadata(
            cfg.dataset.repo_id,
            root=root,
            revision=cfg.dataset.revision,
            repo_type=repo_type,
        )
        if cfg.dataset.feature_keys is not None:
            missing_features = sorted(set(cfg.dataset.feature_keys) - set(ds_meta.features))
            if missing_features:
                raise ValueError(
                    f"Requested dataset.feature_keys are missing from metadata: {missing_features}. "
                    f"Available features: {sorted(ds_meta.features)}"
                )
            retained_keys = [
                key for key in (*cfg.dataset.feature_keys, *DEFAULT_FEATURES) if key in ds_meta.features
            ]
            ds_meta.info.features = {key: ds_meta.features[key] for key in retained_keys}
            if ds_meta.stats is not None:
                ds_meta.stats = {
                    key: value for key, value in ds_meta.stats.items() if key in cfg.dataset.feature_keys
                }
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta, cfg.rename_map)
        selected_episodes = episodes if episodes is not None else cfg.dataset.episodes
        selected_episodes = resolve_episode_indices(
            selected_episodes, ds_meta.total_episodes, cfg.dataset.exclude_episodes
        )
        if cfg.dataset.streaming and ds_meta.storage_format != DEFAULT_STORAGE_FORMAT:
            raise ValueError(
                f"dataset.streaming=True is not supported for storage_format="
                f"{ds_meta.storage_format!r}: StreamingLeRobotDataset only reads the default "
                f"{DEFAULT_STORAGE_FORMAT!r} layout. Note that some formats (e.g. 'lance') "
                "support remote map-style access without streaming mode."
            )
        if not cfg.dataset.streaming:
            if repo_type == "bucket" and ds_meta.storage_format == DEFAULT_STORAGE_FORMAT:
                raise ValueError(
                    f"repo_type='bucket' is streaming-only for the default {DEFAULT_STORAGE_FORMAT!r} "
                    "storage format: set dataset.streaming=true to train from an HF Storage Bucket."
                )
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=selected_episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                tolerance_s=cfg.tolerance_s,
                repo_type=repo_type,
                video_decoder_cache_size=cfg.dataset.video_decoder_cache_size,
                feature_keys=cfg.dataset.feature_keys,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=root,
                episodes=selected_episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                seed=cfg.seed if cfg.seed is not None else DEFAULT_STREAMING_SEED,
                max_num_shards=max(1, cfg.num_workers),
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                video_backend=cfg.dataset.video_backend,
                episode_pool_size=cfg.dataset.streaming_episode_pool_size,
                sampling_strategy=cfg.dataset.streaming_sampling_strategy,
                prefetch_episodes=cfg.dataset.streaming_prefetch_episodes,
                byte_budget_gb=cfg.dataset.streaming_byte_budget_gb,
                decode_threads=cfg.dataset.streaming_decode_threads,
                decoded_queue_size=cfg.dataset.streaming_decoded_queue_size,
                video_decoder_cache_size=cfg.dataset.video_decoder_cache_size,
                native_http_connections=cfg.dataset.streaming_native_http_connections,
                native_http_subranges=cfg.dataset.streaming_native_http_subranges,
                sidecar_lock_timeout_s=cfg.dataset.streaming_sidecar_lock_timeout_s,
                repeat=True,
                repo_type=repo_type,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logger.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            if key in dataset.meta.depth_keys:
                continue  # Exclude depth keys from ImageNet stats
            dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    _apply_normalization_stats(dataset, cfg.dataset.normalization_stats_path)

    return dataset


def _apply_normalization_stats(dataset: LeRobotDataset, stats_path: str | None) -> None:
    """Apply local stats without falling back to potentially leaky dataset-wide stats."""
    if stats_path is None:
        return
    stats_path = Path(stats_path).expanduser()
    if not stats_path.is_file():
        raise FileNotFoundError(f"Normalization statistics file does not exist: {stats_path}")
    custom_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(custom_stats, dict):
        raise ValueError(f"Normalization statistics must be a JSON object: {stats_path}")

    # State and action use MEAN_STD for ACT. Require the supplied train-only values
    # for these features so a missing JSON entry cannot silently fall back to the
    # raw dataset's all-episode metadata statistics.
    required = {key for key in (ACTION, OBS_STATE) if key in dataset.meta.features}
    missing = sorted(required - set(custom_stats))
    if missing:
        raise ValueError(f"Normalization stats are missing required features {missing}: {stats_path}")

    if dataset.meta.stats is None:
        dataset.meta.stats = {}
    for key, feature_stats in custom_stats.items():
        if key not in dataset.meta.features:
            raise ValueError(f"Normalization stats contain unknown feature {key!r}.")
        if not isinstance(feature_stats, dict):
            raise ValueError(f"Normalization stats for {key!r} must be a JSON object.")
        normalized = {
            name: torch.as_tensor(value, dtype=torch.float32) if isinstance(value, list) else value
            for name, value in feature_stats.items()
        }
        if key in required:
            feature_shape = tuple(dataset.meta.features[key]["shape"])
            for name in ("mean", "std"):
                if name not in normalized:
                    raise ValueError(f"Normalization stats for {key!r} must include {name!r}.")
                if not isinstance(normalized[name], torch.Tensor):
                    raise ValueError(f"Normalization {name} for {key!r} must be a numeric vector.")
                if tuple(normalized[name].shape) != feature_shape:
                    raise ValueError(
                        f"Normalization {name} for {key!r} has shape {tuple(normalized[name].shape)}, "
                        f"expected {feature_shape}."
                    )
        dataset.meta.stats[key] = normalized


def make_train_eval_datasets(
    cfg: TrainPipelineConfig,
) -> tuple[LeRobotDataset | StreamingLeRobotDataset, LeRobotDataset | None]:
    """Create train and optional eval datasets from a split manifest or eval_split.

    A manifest may provide an empty validation list; in that case the function returns
    the selected training episodes and ``None`` for the eval dataset.
    """
    if cfg.dataset.split_manifest_path is not None:
        manifest_path = Path(cfg.dataset.split_manifest_path).expanduser()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Episode split manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_episodes = manifest.get("train_episode_indices")
        eval_episodes = manifest.get("validation_episode_indices")
        excluded_episodes = manifest.get("excluded_episode_indices", [])
        if not isinstance(train_episodes, list) or not isinstance(eval_episodes, list):
            raise ValueError(
                "Split manifest must contain train_episode_indices and validation_episode_indices lists."
            )
        if not isinstance(excluded_episodes, list):
            raise ValueError("Split manifest excluded_episode_indices must be a list.")
        for name, indices in (
            ("train_episode_indices", train_episodes),
            ("validation_episode_indices", eval_episodes),
            ("excluded_episode_indices", excluded_episodes),
        ):
            if any(type(index) is not int for index in indices):
                raise ValueError(f"Split manifest {name} must contain only integer episode indices.")
        groups = [set(train_episodes), set(eval_episodes), set(excluded_episodes)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(
                "Train, validation, and excluded episode lists in the manifest must be disjoint."
            )
        if not train_episodes:
            raise ValueError("Split manifest must provide a non-empty training episode list.")
        if cfg.dataset.episodes is not None:
            raise ValueError("Set either dataset.episodes or dataset.split_manifest_path, not both.")
        if not isinstance(cfg.dataset.repo_id, str):
            raise ValueError("dataset.split_manifest_path requires a single dataset repo_id.")
        metadata = load_dataset_metadata(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            revision=cfg.dataset.revision,
            repo_type=cast(Literal["dataset", "bucket"], cfg.dataset.repo_type),
        )
        if manifest.get("codebase_version") != metadata.info.codebase_version:
            raise ValueError(
                "Split manifest codebase_version does not match the dataset: "
                f"{manifest.get('codebase_version')!r} != {metadata.info.codebase_version!r}."
            )
        if manifest.get("fps") != metadata.fps:
            raise ValueError(
                f"Split manifest fps {manifest.get('fps')!r} does not match dataset fps {metadata.fps}."
            )
        declared_root = manifest.get("dataset_root")
        if (
            declared_root is not None
            and Path(declared_root).expanduser().resolve() != metadata.root.resolve()
        ):
            raise ValueError(
                f"Split manifest dataset_root {declared_root!r} does not match dataset root {str(metadata.root)!r}."
            )
        for name, indices in (
            ("train_episode_indices", train_episodes),
            ("validation_episode_indices", eval_episodes),
            ("excluded_episode_indices", excluded_episodes),
        ):
            if not isinstance(indices, list) or any(type(index) is not int for index in indices):
                raise ValueError(f"Split manifest {name} must be a list of integer episode indices.")
            if len(indices) != len(set(indices)):
                raise ValueError(f"Split manifest {name} contains duplicate episode indices.")
            invalid = [index for index in indices if not 0 <= index < metadata.total_episodes]
            if invalid:
                raise ValueError(f"Split manifest {name} has out-of-range indices: {invalid}.")
        logger.info(
            "Using episode split manifest: %d train, %d validation, %d excluded",
            len(train_episodes),
            len(eval_episodes),
            len(excluded_episodes),
        )
        train_dataset = make_dataset(cfg, episodes=train_episodes)
        eval_dataset = make_dataset(cfg, episodes=eval_episodes) if eval_episodes else None
        return train_dataset, eval_dataset

    full_dataset = make_dataset(cfg)

    if cfg.dataset.eval_split == 0.0:
        return full_dataset, None

    base_episodes = (
        full_dataset.episodes if full_dataset.episodes is not None else list(range(full_dataset.num_episodes))
    )

    episode_tasks = full_dataset.meta.episodes["tasks"]
    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx in base_episodes:
        task_key = episode_tasks[ep_idx][0] if episode_tasks[ep_idx] else ""
        task_to_episodes.setdefault(task_key, []).append(ep_idx)

    train_episodes, eval_episodes = [], []
    for eps in task_to_episodes.values():
        n_eval = math.ceil(len(eps) * cfg.dataset.eval_split)
        train_episodes.extend(eps[: len(eps) - n_eval])
        eval_episodes.extend(eps[len(eps) - n_eval :])

    if not train_episodes:
        raise ValueError(
            f"eval_split={cfg.dataset.eval_split} leaves 0 training episodes from {len(base_episodes)} total."
        )

    logger.info(
        f"Train/eval split: {len(train_episodes)} train, {len(eval_episodes)} eval "
        f"(eval_split={cfg.dataset.eval_split}, {len(task_to_episodes)} tasks)"
    )

    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, full_dataset.meta, cfg.rename_map)

    # On the "gpu" backend the workers only decode; lerobot-train augments each batch on the policy device.
    image_transforms_cfg = cfg.dataset.image_transforms
    train_image_transforms = (
        ImageTransforms(image_transforms_cfg)
        if image_transforms_cfg.enable and image_transforms_cfg.backend == "dataloader"
        else None
    )

    train_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=train_image_transforms,
        depth_output_unit=cfg.dataset.depth_output_unit,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        repo_type=cfg.dataset.repo_type,
        video_decoder_cache_size=cfg.dataset.video_decoder_cache_size,
        feature_keys=cfg.dataset.feature_keys,
    )

    eval_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        depth_output_unit=cfg.dataset.depth_output_unit,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        repo_type=cfg.dataset.repo_type,
        video_decoder_cache_size=cfg.dataset.video_decoder_cache_size,
        feature_keys=cfg.dataset.feature_keys,
    )

    if cfg.dataset.use_imagenet_stats:
        for ds in (train_dataset, eval_dataset):
            for key in ds.meta.camera_keys:
                if key in ds.meta.depth_keys:
                    continue
                ds.meta.stats.setdefault(key, {})
                for stats_type, stats in IMAGENET_STATS.items():
                    ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    for ds in (train_dataset, eval_dataset):
        _apply_normalization_stats(ds, cfg.dataset.normalization_stats_path)

    return train_dataset, eval_dataset

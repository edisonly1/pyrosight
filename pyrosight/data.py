"""Data pipeline: TFRecord parsing, normalization, augmentation, PyTorch DataLoader."""

import gzip
import glob
import struct
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config


def _parse_tfrecord_files(file_pattern: str, cfg: Config) -> List[dict]:
    """Parse TFRecord files (plain or gzipped) into a list of numpy dicts.

    Uses a pure-Python parser — no tensorflow dependency required.
    """
    all_keys = cfg.feature_keys + [cfg.label_key]
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for pattern: {file_pattern}")

    samples = []
    for filepath in files:
        for raw in _iter_tfrecord(filepath):
            example = _parse_example(raw, all_keys)
            sample = {}
            for key in all_keys:
                arr = np.array(example[key], dtype=np.float32)
                sample[key] = arr.reshape(cfg.image_size, cfg.image_size)
            samples.append(sample)

    return samples


def _iter_tfrecord(path: str):
    """Yield raw serialized Example bytes from a TFRecord file (plain or gzipped)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        while True:
            # TFRecord format: uint64 length, uint32 crc, data, uint32 crc
            len_bytes = f.read(8)
            if len(len_bytes) < 8:
                break
            length = struct.unpack("<Q", len_bytes)[0]
            f.read(4)  # length CRC
            data = f.read(length)
            f.read(4)  # data CRC
            yield data


def _parse_example(data: bytes, keys: List[str]) -> dict:
    """Parse a tf.train.Example protobuf manually.

    Only handles float_list features (which is all this dataset uses).
    Avoids pulling in protobuf/tensorflow as a dependency.
    """
    # tf.train.Example is: Features { feature: map<string, Feature> }
    # Feature is oneof: bytes_list, float_list, int64_list
    # We parse the protobuf wire format directly.
    result = {}
    pos = 0

    while pos < len(data):
        # Parse top-level field (Features wrapper)
        field_num, wire_type, pos = _read_tag(data, pos)
        if wire_type == 2:  # length-delimited (the Features message)
            length, pos = _read_varint(data, pos)
            features_end = pos + length
            result = _parse_features(data, pos, features_end, keys)
            pos = features_end
        else:
            break

    return result


def _parse_features(data: bytes, start: int, end: int, keys: List[str]) -> dict:
    """Parse the Features message containing the feature map."""
    result = {}
    pos = start

    while pos < end:
        field_num, wire_type, pos = _read_tag(data, pos)
        if wire_type == 2:  # length-delimited (MapEntry)
            length, pos = _read_varint(data, pos)
            entry_end = pos + length
            key, values = _parse_map_entry(data, pos, entry_end)
            if key in keys:
                result[key] = values
            pos = entry_end
        else:
            break

    return result


def _parse_map_entry(data: bytes, start: int, end: int):
    """Parse a single map entry: key (string) -> Feature (float_list)."""
    key = None
    values = []
    pos = start

    while pos < end:
        field_num, wire_type, pos = _read_tag(data, pos)
        if wire_type == 2:  # length-delimited
            length, pos = _read_varint(data, pos)
            field_end = pos + length
            if field_num == 1:  # key (string)
                key = data[pos:field_end].decode("utf-8")
            elif field_num == 2:  # value (Feature message)
                values = _parse_feature(data, pos, field_end)
            pos = field_end
        else:
            break

    return key, values


def _parse_feature(data: bytes, start: int, end: int) -> list:
    """Parse a Feature message, extracting float_list values."""
    pos = start
    while pos < end:
        field_num, wire_type, pos = _read_tag(data, pos)
        if wire_type == 2:  # length-delimited
            length, pos = _read_varint(data, pos)
            field_end = pos + length
            if field_num == 2:  # float_list
                return _parse_float_list(data, pos, field_end)
            pos = field_end
        else:
            break
    return []


def _parse_float_list(data: bytes, start: int, end: int) -> list:
    """Parse FloatList: repeated float packed in a length-delimited field."""
    pos = start
    while pos < end:
        field_num, wire_type, pos = _read_tag(data, pos)
        if wire_type == 2 and field_num == 1:  # packed floats
            length, pos = _read_varint(data, pos)
            n_floats = length // 4
            values = struct.unpack(f"<{n_floats}f", data[pos:pos + length])
            return list(values)
        else:
            break
    return []


def _read_tag(data: bytes, pos: int):
    """Read a protobuf tag (field number + wire type)."""
    tag, pos = _read_varint(data, pos)
    return tag >> 3, tag & 0x7, pos


def _read_varint(data: bytes, pos: int):
    """Read a protobuf varint."""
    result = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


class WildfireDataset(Dataset):
    """PyTorch dataset wrapping parsed TFRecord wildfire data.

    Returns (x_norm, x_raw, label, valid_mask) where:
        x_norm:     (12, 64, 64) float32, z-score normalized (for neural network)
        x_raw:      (12, 64, 64) float32, raw values (for Rothermel physics)
        label:      (64, 64) int64, {0, 1}
        valid_mask: (64, 64) bool, True for known pixels
    """

    def __init__(self, samples: List[dict], cfg: Config, augment: bool = False):
        self.samples = samples
        self.cfg = cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        cfg = self.cfg

        # Stack raw channels (for physics branch — needs physical units)
        raw_channels = []
        # Stack normalized channels (for neural network)
        norm_channels = []

        for i, key in enumerate(cfg.feature_keys):
            ch = sample[key].copy()

            # Replace NaN with 0
            nan_mask = np.isnan(ch)
            if nan_mask.any():
                ch[nan_mask] = 0.0

            raw_channels.append(ch.copy())

            mean, std = cfg.channel_stats[i]
            if key == "PrevFireMask":
                ch = (ch > 0).astype(np.float32)
            else:
                ch = (ch - mean) / (std + 1e-8)
                ch = np.clip(ch, -10.0, 10.0)
            norm_channels.append(ch)

        x_norm = np.stack(norm_channels, axis=0).astype(np.float32)
        x_raw = np.stack(raw_channels, axis=0).astype(np.float32)

        # Label and valid mask
        label_raw = sample[cfg.label_key].copy()
        label_nan = np.isnan(label_raw)
        if label_nan.any():
            label_raw[label_nan] = -1.0
        valid_mask = label_raw >= 0
        label = np.clip(label_raw, 0, 1).astype(np.int64)

        # Convert to tensors
        x_norm = torch.from_numpy(x_norm)
        x_raw = torch.from_numpy(x_raw)
        label = torch.from_numpy(label)
        valid_mask = torch.from_numpy(valid_mask)

        # Augmentation (spatial only — apply same transform to all)
        if self.augment:
            x_norm, x_raw, label, valid_mask = self._augment(x_norm, x_raw, label, valid_mask)

        return x_norm, x_raw, label, valid_mask

    # Wind direction channel index in raw inputs (azimuth in degrees)
    _WIND_DIR_CH = 1

    @staticmethod
    def _augment(x_norm, x_raw, label, mask):
        """Random horizontal flip, vertical flip, and 90° rotation.

        Same spatial transform applied to all tensors. Wind direction
        (channel 1 in x_raw, azimuth degrees 0-360) is corrected to
        maintain physical consistency after each spatial transform.
        """
        hflip = torch.rand(1).item() > 0.5
        vflip = torch.rand(1).item() > 0.5
        k = torch.randint(0, 4, (1,)).item()

        # Apply spatial transforms
        if hflip:
            x_norm = x_norm.flip(-1)
            x_raw = x_raw.flip(-1)
            label = label.flip(-1)
            mask = mask.flip(-1)

        if vflip:
            x_norm = x_norm.flip(-2)
            x_raw = x_raw.flip(-2)
            label = label.flip(-2)
            mask = mask.flip(-2)

        if k > 0:
            x_norm = torch.rot90(x_norm, k, [-2, -1])
            x_raw = torch.rot90(x_raw, k, [-2, -1])
            label = torch.rot90(label, k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])

        # Correct wind direction (azimuth) in x_raw to match spatial transform.
        # Wind azimuth θ: 0°=N, 90°=E, 180°=S, 270°=W.
        # Horizontal flip mirrors east↔west:  θ → (360 - θ) mod 360
        # Vertical flip mirrors north↔south:  θ → (180 - θ) mod 360
        # 90° CCW rotation:                   θ → (θ - 90) mod 360
        ch = WildfireDataset._WIND_DIR_CH
        wind_dir = x_raw[ch]  # (H, W)
        if hflip:
            wind_dir = (-wind_dir) % 360.0
        if vflip:
            wind_dir = (180.0 - wind_dir) % 360.0
        if k > 0:
            wind_dir = (wind_dir - k * 90.0) % 360.0
        x_raw[ch] = wind_dir

        return (x_norm.contiguous(), x_raw.contiguous(),
                label.contiguous(), mask.contiguous())


def _build_fire_sampler(samples: List[dict], label_key: str) -> torch.utils.data.WeightedRandomSampler:
    """Build a sampler that oversamples fire-containing patches 50/50.

    Patches with any fire pixels get upweighted so ~50% of sampled batches
    contain fire, matching the paper's recommendation.
    """
    has_fire = []
    for s in samples:
        label = s[label_key]
        has_fire.append((label == 1).any())

    n_fire = sum(has_fire)
    n_nofire = len(has_fire) - n_fire
    print(f"  Fire-containing patches: {n_fire}/{len(has_fire)} ({100*n_fire/len(has_fire):.1f}%)")

    # Weight so fire and no-fire patches are sampled equally
    w_fire_patch = 1.0 / max(n_fire, 1)
    w_nofire_patch = 1.0 / max(n_nofire, 1)
    weights = [w_fire_patch if hf else w_nofire_patch for hf in has_fire]

    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(samples), replacement=True,
    )


def build_dataloaders(
    cfg: Config, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Parse TFRecords and return train/val/test DataLoaders."""
    print("Parsing training data...")
    train_samples = _parse_tfrecord_files(cfg.train_pattern, cfg)
    print(f"  {len(train_samples)} training samples")

    print("Parsing validation data...")
    val_samples = _parse_tfrecord_files(cfg.val_pattern, cfg)
    print(f"  {len(val_samples)} validation samples")

    print("Parsing test data...")
    test_samples = _parse_tfrecord_files(cfg.test_pattern, cfg)
    print(f"  {len(test_samples)} test samples")

    train_ds = WildfireDataset(train_samples, cfg, augment=True)
    val_ds = WildfireDataset(val_samples, cfg, augment=False)
    test_ds = WildfireDataset(test_samples, cfg, augment=False)

    # Check fire patch prevalence to decide sampling strategy
    n_fire_patches = sum(1 for s in train_samples if (s[cfg.label_key] == 1).any())
    fire_frac = n_fire_patches / len(train_samples)
    print(f"  Fire-containing patches: {n_fire_patches}/{len(train_samples)} ({100*fire_frac:.1f}%)")

    if fire_frac < 0.4:
        # Fire patches are rare — oversample them to 50/50
        print("  Using fire-patch oversampling")
        train_sampler = _build_fire_sampler(train_samples, cfg.label_key)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=False, drop_last=True,
        )
    else:
        # Fire patches already prevalent — just shuffle
        print("  Fire patches prevalent, using standard shuffle")
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False, drop_last=True,
        )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader, test_loader

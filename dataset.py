import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0

    def transform(self, data: np.ndarray):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray):
        return (data * self.std) + self.mean


class MaxAbsScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        scale = np.max(np.abs(data), axis=0, keepdims=True)
        scale[scale == 0] = 1.0
        self.mean = np.zeros((1, data.shape[1]), dtype=np.float32)
        self.std = scale.astype(np.float32)

    def fit_global(self, data: np.ndarray):
        scale = float(np.max(np.abs(data)))
        if scale == 0:
            scale = 1.0
        self.mean = np.zeros((1, data.shape[1]), dtype=np.float32)
        self.std = np.full((1, data.shape[1]), scale, dtype=np.float32)

    def transform(self, data: np.ndarray):
        return data / self.std

    def inverse_transform(self, data: np.ndarray):
        return data * self.std


def build_time_features(index: pd.DatetimeIndex, freq: str, timeenc: int) -> np.ndarray:
    freq_lower = freq.lower()

    if timeenc == 0:
        data = {
            "month": index.month.astype(np.float32),
            "day": index.day.astype(np.float32),
            "weekday": index.dayofweek.astype(np.float32),
            "hour": index.hour.astype(np.float32),
        }
        if "min" in freq_lower or freq_lower.endswith("t"):
            data["minute"] = (index.minute // 15).astype(np.float32)
        return np.column_stack(list(data.values())).astype(np.float32)

    features = []
    if "min" in freq_lower or freq_lower.endswith("t"):
        features.append(index.minute / 59.0 - 0.5)
    if freq_lower.endswith("h") or "min" in freq_lower or freq_lower.endswith("t"):
        features.append(index.hour / 23.0 - 0.5)
    if freq_lower.endswith("d") or freq_lower.endswith("h") or "min" in freq_lower or freq_lower.endswith("t"):
        features.append(index.dayofweek / 6.0 - 0.5)
        features.append((index.day - 1) / 30.0 - 0.5)
        features.append((index.dayofyear - 1) / 365.0 - 0.5)
    if freq_lower.endswith("w") or freq_lower.endswith("m"):
        features.append((index.month - 1) / 11.0 - 0.5)


    return np.vstack(features).T.astype(np.float32)


@dataclass
class SplitBorders:
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def compute_ratio_borders(
    length: int,
    seq_len: int,
    pred_len: int,
    forecast_horizon: int,
    train_ratio: float,
    val_ratio: float,
):
    test_ratio = 1.0 - train_ratio - val_ratio

    num_train = int(length * train_ratio)
    num_test = int(length * test_ratio)
    num_val = length - num_train - num_test

    border1s = [0, num_train - seq_len, length - num_test - seq_len]
    border2s = [num_train, num_train + num_val, length]
    return border1s, border2s


def build_dummy_time_features(length: int) -> np.ndarray:
    return np.zeros((length, 1), dtype=np.float32)


class ETTHourDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        data_path: str = "ETTh1.csv",
        flag: str = "train",
        size=None,
        features: str = "M",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 1,
        freq: str = "h",
        forecast_horizon: int = 1,
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.pred_len = size

        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.forecast_horizon = forecast_horizon
        self.root_path = root_path
        self.data_path = data_path

        self.scaler = StandardScaler()
        self.feature_names = []
        self.split_borders = None
        self.data_x = None
        self.data_y = None
        self.data_stamp = None
        self.wind_metadata = None

        self._read_data()

    def _read_csv(self) -> pd.DataFrame:
        csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(csv_path)
        return df_raw

    def _read_data(self) -> None:
        df_raw = self._read_csv()

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        split_map = {"train": 0, "val": 1, "test": 2}
        split_index = split_map[self.flag]
        border1 = border1s[split_index]
        border2 = border2s[split_index]

        self.split_borders = SplitBorders(
            train_start=border1s[0],
            train_end=border2s[0],
            val_start=border1s[1],
            val_end=border2s[1],
            test_start=border1s[2],
            test_end=border2s[2],
        )

        all_feature_columns = [column for column in df_raw.columns if column != "date"]
        if self.features == "M":
            feature_columns = all_feature_columns
        elif self.features == "MS":
            non_target = [column for column in all_feature_columns if column != self.target]
            feature_columns = non_target + [self.target]
        elif self.features == "S":
            feature_columns = [self.target]
        else:
            raise ValueError("features must be one of: M, S, MS")

        self.feature_names = feature_columns
        df_data = df_raw[feature_columns]

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]].values.astype(np.float32)
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values.astype(np.float32))
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_raw.loc[border1:border2 - 1, ["date"]].copy()
        date_index = pd.to_datetime(df_stamp["date"].values)
        data_stamp = build_time_features(date_index, freq=self.freq, timeenc=self.timeenc)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - (self.forecast_horizon - 1) - self.pred_len + 1

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        future_begin = s_end + self.forecast_horizon - 1
        future_end = future_begin + self.pred_len

        seq_x = torch.from_numpy(self.data_x[s_begin:s_end]).float()
        seq_x_mark = torch.from_numpy(self.data_stamp[s_begin:s_end]).float()
        seq_y = torch.from_numpy(self.data_y[future_begin:future_end]).float()

        return seq_x, seq_y, seq_x_mark

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


class CustomCSVDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: str = "train",
        size=None,
        features: str = "MS",
        target: str = "Target",
        time_col: str = "DATE",
        scale: bool = True,
        timeenc: int = 1,
        freq: str = "h",
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        replace_zero_as_nan: bool = False,
        interpolate_missing: bool = False,
        forecast_horizon: int = 1,
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.pred_len = size


        self.flag = flag
        self.features = features
        self.target = target
        self.time_col = time_col
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.replace_zero_as_nan = replace_zero_as_nan
        self.interpolate_missing = interpolate_missing
        self.forecast_horizon = forecast_horizon

        self.scaler = StandardScaler()
        self.feature_names = []
        self.split_borders = None
        self.data_x = None
        self.data_y = None
        self.data_stamp = None
        self.wind_metadata = None

        self._read_data()

    def _read_csv(self) -> pd.DataFrame:
        csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(csv_path)
        return df_raw

    def _read_data(self) -> None:
        df_raw = self._read_csv()
        df_raw[self.time_col] = pd.to_datetime(df_raw[self.time_col])
        df_raw = df_raw.sort_values(self.time_col).drop_duplicates(self.time_col)

        all_feature_columns = [column for column in df_raw.columns if column != self.time_col]
        if self.features == "M":
            feature_columns = all_feature_columns
        elif self.features == "MS":
            non_target = [column for column in all_feature_columns if column != self.target]
            feature_columns = non_target + [self.target]
        elif self.features == "S":
            feature_columns = [self.target]
        else:
            raise ValueError("features must be: M, S, MS")

        self.feature_names = feature_columns
        df_data = df_raw[feature_columns].astype(np.float32)

        if self.replace_zero_as_nan:
            df_data = df_data.replace(0, np.nan)
        if self.interpolate_missing:
            df_data = df_data.interpolate(method="linear").ffill().bfill()

        border1s, border2s = compute_ratio_borders(
            length=len(df_raw),
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            forecast_horizon=self.forecast_horizon,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )
        split_map = {"train": 0, "val": 1, "test": 2}
        split_index = split_map[self.flag]
        border1 = border1s[split_index]
        border2 = border2s[split_index]

        self.split_borders = SplitBorders(
            train_start=border1s[0],
            train_end=border2s[0],
            val_start=border1s[1],
            val_end=border2s[1],
            test_start=border1s[2],
            test_end=border2s[2],
        )

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]].values.astype(np.float32)
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values.astype(np.float32))
        else:
            data = df_data.values.astype(np.float32)

        df_stamp = df_raw.iloc[border1:border2][[self.time_col]].copy()
        date_index = pd.to_datetime(df_stamp[self.time_col].values)
        data_stamp = build_time_features(date_index, freq=self.freq, timeenc=self.timeenc)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - (self.forecast_horizon - 1) - self.pred_len + 1

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        future_begin = s_end + self.forecast_horizon - 1
        future_end = future_begin + self.pred_len

        seq_x = torch.from_numpy(self.data_x[s_begin:s_end]).float()
        seq_x_mark = torch.from_numpy(self.data_stamp[s_begin:s_end]).float()
        seq_y = torch.from_numpy(self.data_y[future_begin:future_end]).float()

        return seq_x, seq_y, seq_x_mark

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


class MTGNNTextDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: str = "train",
        size=None,
        features: str = "M",
        target: str = "var_0",
        scale: bool = True,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        normalize: int = 2,
        forecast_horizon: int = 1,
    ):
        if size is None:
            self.seq_len = 24 * 7
            self.pred_len = 1
        else:
            self.seq_len, self.pred_len = size

        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.normalize = normalize
        self.forecast_horizon = forecast_horizon

        self.scaler = MaxAbsScaler()
        self.feature_names = []
        self.split_borders = None
        self.data_x = None
        self.data_y = None
        self.data_stamp = None
        self.sample_indices = None
        self.wind_metadata = None

        self._read_data()

    def _read_file(self) -> np.ndarray:
        txt_path = os.path.join(self.root_path, self.data_path)
        data = np.loadtxt(txt_path, delimiter=",", dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data

    def _read_data(self) -> None:
        raw_data = self._read_file()
        all_feature_columns = [f"var_{index}" for index in range(raw_data.shape[1])]

        if self.features == "M":
            feature_indices = list(range(len(all_feature_columns)))
        elif self.features == "MS":
            target_index = all_feature_columns.index(self.target)
            feature_indices = [index for index in range(len(all_feature_columns)) if index != target_index] + [target_index]
        elif self.features == "S":
            feature_indices = [all_feature_columns.index(self.target)]
        else:
            raise ValueError("features must be: M, S, MS")

        self.feature_names = [all_feature_columns[index] for index in feature_indices]
        data_selected = raw_data[:, feature_indices].astype(np.float32)

        if self.scale:
            if self.normalize == 2:
                self.scaler.fit(data_selected)
            elif self.normalize == 1:
                self.scaler.fit_global(data_selected)
            elif self.normalize == 0:
                self.scaler.mean = np.zeros((1, data_selected.shape[1]), dtype=np.float32)
                self.scaler.std = np.ones((1, data_selected.shape[1]), dtype=np.float32)
            else:
                raise ValueError("normalize must be: 0, 1, 2")
            data = self.scaler.transform(data_selected)
        else:
            self.scaler.mean = np.zeros((1, data_selected.shape[1]), dtype=np.float32)
            self.scaler.std = np.ones((1, data_selected.shape[1]), dtype=np.float32)
            data = data_selected

        total_length = len(data)
        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.val_ratio))

        split_map = {
            "train": np.arange(self.seq_len + self.forecast_horizon - 1, train_end - self.pred_len + 1),
            "val": np.arange(train_end, val_end - self.pred_len + 1),
            "test": np.arange(val_end, total_length - self.pred_len + 1),
        }
        self.sample_indices = split_map[self.flag]
        self.split_borders = SplitBorders(
            train_start=0,
            train_end=train_end,
            val_start=train_end,
            val_end=val_end,
            test_start=val_end,
            test_end=total_length,
        )

        self.data_x = data
        self.data_y = data
        self.data_stamp = build_dummy_time_features(total_length)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int):
        future_begin = int(self.sample_indices[index])
        history_end = future_begin - self.forecast_horizon + 1
        history_begin = history_end - self.seq_len
        future_end = future_begin + self.pred_len

        seq_x = torch.from_numpy(self.data_x[history_begin:history_end]).float()
        seq_x_mark = torch.from_numpy(self.data_stamp[history_begin:history_end]).float()
        seq_y = torch.from_numpy(self.data_y[future_begin:future_end]).float()
        return seq_x, seq_y, seq_x_mark

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


def detect_wind_node_mapping(columns):
    pattern = re.compile(r"(.+)_WT(\d+)$")
    rows = []
    for column in columns:
        match = pattern.match(str(column))
        if match is None:
            continue
        rows.append(
            {
                "column": str(column),
                "feature": match.group(1),
                "node": int(match.group(2)),
            }
        )
    return pd.DataFrame(rows)


class WindSTIDDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: str = "train",
        size=None,
        target: str = "Target",
        time_col: str = "DATE",
        scale: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        replace_zero_as_nan: bool = False,
        interpolate_missing: bool = False,
        forecast_horizon: int = 1,
        train_noise_std: float = 0.0,
    ):
        if size is None:
            self.seq_len = 72
            self.pred_len = 1
        else:
            self.seq_len, self.pred_len = size

        self.flag = flag
        self.target = target
        self.time_col = time_col
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.replace_zero_as_nan = replace_zero_as_nan
        self.interpolate_missing = interpolate_missing
        self.forecast_horizon = forecast_horizon
        self.train_noise_std = train_noise_std if flag == "train" else 0.0

        self.scaler = StandardScaler()
        self.node_mean = None
        self.node_std = None
        self.feature_names = [target]
        self.raw_feature_names = [target]
        self.node_feature_names = []
        self.node_ids = []
        self.split_borders = None
        self.data_x = None
        self.data_y = None
        self.data_stamp = None
        self.wind_metadata = None

        self._read_data()

    def _read_csv(self) -> pd.DataFrame:
        csv_path = os.path.join(self.root_path, self.data_path)
        return pd.read_csv(csv_path)

    def _read_data(self) -> None:
        df_raw = self._read_csv()
        df_raw.columns = [str(column).strip() for column in df_raw.columns]

        if self.time_col in df_raw.columns:
            df_raw[self.time_col] = pd.to_datetime(df_raw[self.time_col], errors="coerce", utc=True)
            df_raw = df_raw.sort_values(self.time_col).drop_duplicates(self.time_col).reset_index(drop=True)

        
        mapping = detect_wind_node_mapping(df_raw.columns)

        self.node_ids = sorted(mapping["node"].unique().tolist())
        features_by_node = {
            node: set(mapping.loc[mapping["node"] == node, "feature"])
            for node in self.node_ids
        }
        self.node_feature_names = sorted(set.intersection(*features_by_node.values()))


        for column in df_raw.columns:
            if column != self.time_col:
                df_raw[column] = pd.to_numeric(df_raw[column], errors="coerce")

        node_columns = [
            f"{feature}_WT{node}"
            for node in self.node_ids
            for feature in self.node_feature_names
        ]
        df_values = df_raw[node_columns + [self.target]].copy()

        if self.replace_zero_as_nan:
            df_values = df_values.replace(0, np.nan)
        if self.interpolate_missing:
            df_values = df_values.interpolate(method="linear").ffill().bfill()
        df_values = df_values.dropna().reset_index(drop=True)

        node_data = np.empty(
            (len(df_values), len(self.node_ids), len(self.node_feature_names)),
            dtype=np.float32,
        )
        for node_index, node in enumerate(self.node_ids):
            for feature_index, feature in enumerate(self.node_feature_names):
                node_data[:, node_index, feature_index] = df_values[f"{feature}_WT{node}"].to_numpy(dtype=np.float32)

        target_data = df_values[self.target].to_numpy(dtype=np.float32).reshape(-1, 1)

        border1s, border2s = compute_ratio_borders(
            length=len(df_values),
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            forecast_horizon=self.forecast_horizon,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )
        split_map = {"train": 0, "val": 1, "test": 2}
        split_index = split_map[self.flag]
        border1 = border1s[split_index]
        border2 = border2s[split_index]

        self.split_borders = SplitBorders(
            train_start=border1s[0],
            train_end=border2s[0],
            val_start=border1s[1],
            val_end=border2s[1],
            test_start=border1s[2],
            test_end=border2s[2],
        )

        if self.scale:
            train_node = node_data[border1s[0]:border2s[0]]
            self.node_mean = train_node.mean(axis=0, keepdims=True)
            self.node_std = train_node.std(axis=0, keepdims=True)
            self.node_std[self.node_std == 0] = 1.0
            node_data = (node_data - self.node_mean) / self.node_std

            self.scaler.fit(target_data[border1s[0]:border2s[0]])
            target_data = self.scaler.transform(target_data)
        else:
            self.node_mean = np.zeros((1, len(self.node_ids), len(self.node_feature_names)), dtype=np.float32)
            self.node_std = np.ones((1, len(self.node_ids), len(self.node_feature_names)), dtype=np.float32)
            self.scaler.mean = np.zeros((1, 1), dtype=np.float32)
            self.scaler.std = np.ones((1, 1), dtype=np.float32)

        self.data_x = node_data[border1:border2].astype(np.float32)
        self.data_y = target_data[border1:border2].astype(np.float32)
        self.data_stamp = build_dummy_time_features(len(self.data_x))
        self.wind_metadata = {
            "node_ids": self.node_ids,
            "node_feature_names": self.node_feature_names,
            "node_data_shape": list(self.data_x.shape),
        }

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - (self.forecast_horizon - 1) - self.pred_len + 1

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        future_begin = s_end + self.forecast_horizon - 1
        future_end = future_begin + self.pred_len

        seq_x = torch.from_numpy(self.data_x[s_begin:s_end]).float()
        if self.train_noise_std > 0:
            seq_x = seq_x + torch.randn_like(seq_x) * self.train_noise_std

        seq_x_mark = torch.from_numpy(self.data_stamp[s_begin:s_end]).float()
        seq_y = torch.from_numpy(self.data_y[future_begin:future_end]).float()
        return seq_x, seq_y, seq_x_mark

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 4:
            return (data * self.node_std) + self.node_mean
        return self.scaler.inverse_transform(data)

    @property
    def num_features(self) -> int:
        return len(self.node_feature_names)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)


def create_data_loaders(args):
    timeenc = 1 if args.embed == "timeF" else 0
    data_name = args.data.lower()

    if data_name in {"etth1"}:
        dataset_class = ETTHourDataset
        dataset_kwargs = dict(
            root_path=args.root_path,
            data_path=args.data_path,
            size=[args.seq_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=not args.no_scale,
            timeenc=timeenc,
            freq=args.freq,
            forecast_horizon=args.forecast_horizon,
        )
    elif data_name in {"custom", "combined"}:
        dataset_class = CustomCSVDataset
        dataset_kwargs = dict(
            root_path=args.root_path,
            data_path=args.data_path,
            size=[args.seq_len, args.pred_len],
            features=args.features,
            target=args.target,
            time_col=args.time_col,
            scale=not args.no_scale,
            timeenc=timeenc,
            freq=args.freq,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            replace_zero_as_nan=args.replace_zero_as_nan,
            interpolate_missing=args.interpolate_missing,
            forecast_horizon=args.forecast_horizon,
        )
    elif data_name in {"mtgnn_txt", "electricity_mtgnn"}:
        dataset_class = MTGNNTextDataset
        train_ratio = 0.6 if data_name != "mtgnn_txt" else args.train_ratio
        val_ratio = 0.2 if data_name != "mtgnn_txt" else args.val_ratio
        dataset_kwargs = dict(
            root_path=args.root_path,
            data_path=args.data_path,
            size=[args.seq_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=not args.no_scale,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            normalize=args.txt_normalize,
            forecast_horizon=args.forecast_horizon,
        )
    elif data_name in {"wind_stid", "stid_ensemble", "xgboost_stid"}:
        dataset_class = WindSTIDDataset
        dataset_kwargs = dict(
            root_path=args.root_path,
            data_path=args.data_path,
            size=[args.seq_len, args.pred_len],
            target=args.target,
            time_col=args.time_col,
            scale=not args.no_scale,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            replace_zero_as_nan=args.replace_zero_as_nan,
            interpolate_missing=args.interpolate_missing,
            forecast_horizon=args.forecast_horizon,
            train_noise_std=args.stid_train_noise_std,
        )


    train_dataset = dataset_class(flag="train", **dataset_kwargs)
    val_dataset = dataset_class(flag="val", **dataset_kwargs)
    test_dataset = dataset_class(flag="test", **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader

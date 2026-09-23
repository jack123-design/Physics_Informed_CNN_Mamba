import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class VRDLRDataset(Dataset):
    def __init__(self, base_dir, subjects, window_size=30, rms_window=100, rms_stride=10, is_train=True, scaler=None):
        self.base_dir = base_dir
        self.window_size = window_size
        self.rms_window = rms_window
        self.rms_stride = rms_stride
        self.is_train = is_train

        self.emg_cols = ["R-IL[Raw](V)", "L-IL[Raw](V)", "L-LO[Raw](V)", "R-LO[Raw](V)"]
        self.coord_all = ["C7[x](cm)", "C7[y](cm)", "C7[z](cm)", "L5[x](cm)", "L5[y](cm)", "L5[z](cm)"]

        # 20-450Hz 带通滤波器 (1000Hz 采样率)
        self.b, self.a = butter(4, [20.0 / 500.0, 450.0 / 500.0], btype='band')

        self.data_windows = []
        self.targets = []
        self.scaler = scaler if scaler else StandardScaler()

        self._load_data(subjects)

    def _bandpass_filter(self, signal):
        return filtfilt(self.b, self.a, signal, axis=0)

    def _compute_rms(self, signal):
        n_samples, n_channels = signal.shape
        n_windows = (n_samples - self.rms_window) // self.rms_stride + 1
        if n_windows <= 0:
            return np.zeros((0, n_channels))

        rms_values = np.zeros((n_windows, n_channels))
        for i in range(n_windows):
            start = i * self.rms_stride
            end = start + self.rms_window
            rms_values[i] = np.sqrt(np.mean(signal[start:end] ** 2, axis=0))
        return rms_values

    def _load_data(self, subjects):
        all_csvs = []
        print(f"[路径调试] 基础数据目录: {self.base_dir}")

        for sub in subjects:
            sub_dir = os.path.join(self.base_dir, sub)
            pattern = os.path.join(sub_dir, "**", "*.csv")
            found = glob.glob(pattern, recursive=True)
            all_csvs.extend(found)

        all_csvs = [f for f in all_csvs if "MVC" not in f]
        print(f"[调试] 最终有效处理文件总数: {len(all_csvs)}")

        raw_features_list = []
        raw_targets_list = []

        # 强制测试前 3 个文件，直接暴露出所有细节
        for idx, file in enumerate(all_csvs[:3]):
            print(f"\n--- 正在严格剖析第 {idx + 1} 个文件: {os.path.basename(file)} ---")
            df = pd.read_csv(file, encoding="gbk")
            df.columns = df.columns.str.strip().str.replace('\r', '').str.replace('\n', '')

            # 强制全部转数值，防止带有不可见字符
            for col in self.emg_cols + self.coord_all:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 【核心排查与修复】：肌电先结束，动作后结束！
            # 找到肌电数据的最后一行有效索引，直接截断后面的空白行，坚决不喂假数据！
            valid_emg_idx = df[self.emg_cols].dropna(how='all').index.max()
            if pd.isna(valid_emg_idx):
                print("  ! 该文件无肌电数据，跳过")
                continue
            df = df.loc[:valid_emg_idx].copy()

            # 【插值修复 1】：动作数据是 0.1s 间隔，有大量内部空洞，进行线性插值对齐到 0.01s
            for col in self.coord_all:
                if col in df.columns:
                    df[col] = df[col].interpolate(method='linear').bfill().ffill()

            # 【插值修复 2】：肌电内部可能存在的零星断点保护
            for col in self.emg_cols:
                df[col] = df[col].interpolate(method='linear').bfill().ffill().fillna(0.0)

            emg_raw = df[self.emg_cols].values
            c7_x, c7_y, c7_z = df["C7[x](cm)"].values, df["C7[y](cm)"].values, df["C7[z](cm)"].values
            l5_x, l5_y, l5_z = df["L5[x](cm)"].values, df["L5[y](cm)"].values, df["L5[z](cm)"].values
            print(f"  1. 原始数据读取与截断成功 | EMG形状: {emg_raw.shape}, L5长度: {len(l5_y)}")

            # 动态解算标签 (保留你极其出色的矢状面几何计算)
            dy = c7_y - l5_y
            dz = c7_z - l5_z
            raw_angle = np.degrees(np.arctan2(dy, np.abs(dz)))

            neutral_samples = min(500, len(raw_angle))
            neutral_angle = np.median(raw_angle[:neutral_samples])
            lumbar_angle = raw_angle - neutral_angle
            lumbar_angle = np.clip(lumbar_angle, -90.0, 90.0)

            lumbar_torque = 45.0 * 9.81 * 0.30 * np.sin(np.deg2rad(lumbar_angle))
            target_raw = np.column_stack((lumbar_angle, lumbar_torque))
            target_raw = np.nan_to_num(target_raw, nan=0.0)
            print(f"  2. 物理标签解算成功 | 目标矩阵形状: {target_raw.shape}")

            emg_filtered = np.abs(self._bandpass_filter(emg_raw))
            emg_rms = self._compute_rms(emg_filtered)
            print(f"  3. 带通滤波与 RMS 降频成功 | RMS 特征形状: {emg_rms.shape}")

            target_downsampled = target_raw[self.rms_window - 1:: self.rms_stride][:len(emg_rms), :]
            print(f"  4. 标签对齐成功 | 对齐后标签形状: {target_downsampled.shape}")
            print(f"  5. 窗口大小: {self.window_size} | 实际RMS长度: {len(emg_rms)}")

        # 正常跑完前 3 个文件的诊断后，执行全量循环
        for file in all_csvs:
            df = pd.read_csv(file, encoding="gbk")
            df.columns = df.columns.str.strip().str.replace('\r', '').str.replace('\n', '')

            for col in self.emg_cols + self.coord_all:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            valid_emg_idx = df[self.emg_cols].dropna(how='all').index.max()
            if pd.isna(valid_emg_idx):
                continue
            df = df.loc[:valid_emg_idx].copy()

            for col in self.coord_all:
                if col in df.columns:
                    df[col] = df[col].interpolate(method='linear').bfill().ffill()

            for col in self.emg_cols:
                df[col] = df[col].interpolate(method='linear').bfill().ffill().fillna(0.0)

            emg_raw = df[self.emg_cols].values
            c7_y, c7_z = df["C7[y](cm)"].values, df["C7[z](cm)"].values
            l5_y, l5_z = df["L5[y](cm)"].values, df["L5[z](cm)"].values

            dy = c7_y - l5_y
            dz = c7_z - l5_z
            raw_angle = np.degrees(np.arctan2(dy, np.abs(dz)))

            neutral_samples = min(500, len(raw_angle))
            neutral_angle = np.median(raw_angle[:neutral_samples])
            lumbar_angle = raw_angle - neutral_angle
            lumbar_angle = np.clip(lumbar_angle, -90.0, 90.0)

            lumbar_torque = 45.0 * 9.81 * 0.30 * np.sin(np.deg2rad(lumbar_angle))
            target_raw = np.column_stack((lumbar_angle, lumbar_torque))
            target_raw = np.nan_to_num(target_raw, nan=0.0)

            emg_filtered = np.abs(self._bandpass_filter(emg_raw))
            emg_rms = self._compute_rms(emg_filtered)
            target_downsampled = target_raw[self.rms_window - 1:: self.rms_stride][:len(emg_rms), :]

            if len(emg_rms) >= self.window_size:
                raw_features_list.append(emg_rms)
                raw_targets_list.append(target_downsampled)

        print(f"[调试] 成功通过长度筛选的原始文件片段数: {len(raw_features_list)}")
        if not raw_features_list:
            raise ValueError("【诊断中止】所有文件均无效！")

        X_concat = np.vstack(raw_features_list)
        if self.is_train:
            self.scaler.fit(X_concat)
            if hasattr(self.scaler, 'scale_'):
                self.scaler.scale_[self.scaler.scale_ == 0] = 1e-5

        for i in range(len(raw_features_list)):
            norm_emg = self.scaler.transform(raw_features_list[i])
            norm_emg = np.nan_to_num(norm_emg, nan=0.0, posinf=1.0, neginf=-1.0)
            y_arr = raw_targets_list[i]

            for j in range(len(norm_emg) - self.window_size + 1):
                self.data_windows.append(norm_emg[j: j + self.window_size])
                self.targets.append(y_arr[j + self.window_size - 1])

        self.data_windows = np.array(self.data_windows, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        print(f"[调试] 最终成功构建的滑动窗口样本总数: {len(self.data_windows)}")

    def __len__(self):
        return len(self.data_windows)

    def __getitem__(self, idx):
        return torch.tensor(self.data_windows[idx]), torch.tensor(self.targets[idx])
# -*- coding: utf-8 -*-


import os
import sys
import csv
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# 外部依赖
from pydub import AudioSegment
import librosa
import soundfile as sf


TARGET_SR = 22050

# 过滤阈值（仅在 use_feature_filtering=True 时生效）
MIN_DURATION = 1.0   # s
MAX_DURATION = 10.0  # s
RMS_MIN = 0.06
RMS_MAX = 0.18
ZCR_MAX = 0.10
SPEC_MEAN_MIN = 0.25
SPEC_STD_MIN  = 1.00

# 静音处理与幅度控制
SILENCE_DB = -40         # librosa 的 top_db 使用正数，这里取绝对值
MAX_SILENCE_KEEP = 0.8   # s，内部静音段最多保留
MAX_PEAK = 0.90          # 峰值硬限幅（避免剪裁）
TARGET_RMS = 0.12        # 目标 RMS（幅度归一化）

# 文件扫描
EXTS = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]




def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_audio_files(in_dir: str, exts: Optional[List[str]] = None, recursive: bool = False) -> List[str]:
    if exts is None:
        exts = EXTS
    exts = set([e.lower() for e in exts])
    files = []
    if recursive:
        for root, _, fnames in os.walk(in_dir):
            for fn in fnames:
                if fn.startswith("."):  # 跳过隐藏文件
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts:
                    files.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(in_dir):
            p = os.path.join(in_dir, fn)
            if not os.path.isfile(p) or fn.startswith("."):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                files.append(p)
    return sorted(files)


def load_audio_anyformat(path: str, target_sr: int = TARGET_SR) -> Tuple[Optional[np.ndarray], Optional[int]]:
    try:
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(target_sr)
        arr = np.array(audio.get_array_of_samples())
        # 防止整数转换溢出
        denom = float(1 << (8 * audio.sample_width - 1))
        y = (arr.astype(np.float32) / denom).clip(-1.0, 1.0)
        return y, target_sr
    except Exception as e:
        print(f"[ERROR] load_audio_anyformat failed: {path}, err={e}")
        return None, None


def trim_silence(y: np.ndarray, sr: int, top_db: float = SILENCE_DB) -> np.ndarray:
    if y.size == 0:
        return y
    try:
        yt, _ = librosa.effects.trim(y, top_db=abs(top_db))
        return yt if yt.size > 0 else y
    except Exception:
        return y


def compress_internal_silence(y: np.ndarray, sr: int,
                              top_db: float = SILENCE_DB,
                              max_silence: float = MAX_SILENCE_KEEP) -> np.ndarray:
    if y.size == 0:
        return y
    try:
        non_sil = librosa.effects.split(y, top_db=abs(top_db), frame_length=1024, hop_length=256)
    except Exception:
        return y
    if non_sil is None or len(non_sil) == 0:
        # 全静音，返回原音频
        return y
    out = []
    prev_end = 0
    max_keep = int(max_silence * sr)
    for start, end in non_sil:
        if start > prev_end:
            sil_len = start - prev_end
            keep = min(sil_len, max_keep)
            if keep > 0:
                out.append(y[prev_end:prev_end + keep])
        out.append(y[start:end])
        prev_end = end

    if prev_end < len(y):
        sil_len = len(y) - prev_end
        keep = min(sil_len, max_keep)
        if keep > 0:
            out.append(y[prev_end:prev_end + keep])
    return np.concatenate(out) if out else y


def rms_normalize(y: np.ndarray, target_rms: float = TARGET_RMS) -> Tuple[np.ndarray, float]:
    if y.size == 0:
        return y, 0.0
    rms = float(np.sqrt(np.mean(y**2))) if y.size > 0 else 0.0
    if rms <= 1e-12:
        return y, rms  # 完全静音，无需处理

    peak = float(np.max(np.abs(y))) if y.size > 0 else 0.0
    if peak > MAX_PEAK and peak > 0:
        y = (y * (MAX_PEAK / peak)).clip(-1.0, 1.0)
        rms = float(np.sqrt(np.mean(y**2)))

    if rms < 0.01:
        gain = min(target_rms / max(rms, 1e-8), 10.0)
    elif rms > 0.25:
        gain = target_rms / rms
    else:
        return y, rms
    y_norm = (y * gain).clip(-1.0, 1.0)
    return y_norm, float(np.sqrt(np.mean(y_norm**2)))


def compute_freq_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    if y.size == 0:
        return {"spec_mean": 0.0, "spec_std": 0.0, "mid_mean": 0.0, "hi_mean": 0.0, "low_mean": 0.0}
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    if S.size == 0:
        return {"spec_mean": 0.0, "spec_std": 0.0, "mid_mean": 0.0, "hi_mean": 0.0, "low_mean": 0.0}
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mean = float(np.mean(S))
    std = float(np.std(S))
    mid_band = (freqs >= 300) & (freqs <= 3000)
    hi_band = (freqs > 3000) & (freqs <= 8000)
    low_band = (freqs < 300)
    mid_mean = float(S[mid_band, :].mean()) if np.any(mid_band) else 0.0
    hi_mean  = float(S[hi_band, :].mean()) if np.any(hi_band) else 0.0
    low_mean = float(S[low_band, :].mean()) if np.any(low_band) else 0.0
    return {"spec_mean": mean, "spec_std": std, "mid_mean": mid_mean, "hi_mean": hi_mean, "low_mean": low_mean}


def zero_crossing_rate(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    zc = librosa.zero_crossings(y, pad=False)
    return float(np.mean(zc)) if zc.size > 0 else 0.0



def process_audio_for_ablation(path: str, out_dir: str, config: Dict) -> List[Dict]:

    y, sr = load_audio_anyformat(path, target_sr=TARGET_SR)
    if y is None:
        return []

    orig_len = len(y) / sr if sr else 0.0


    if config.get("use_trim_silence", False):
        y = trim_silence(y, sr, top_db=SILENCE_DB)
    if config.get("use_compress_internal_silence", False):
        y = compress_internal_silence(y, sr, top_db=SILENCE_DB, max_silence=MAX_SILENCE_KEEP)


    if config.get("use_feature_filtering", False):
        dur = len(y) / sr if sr else 0.0
        if dur < MIN_DURATION or dur > MAX_DURATION:
            return []
        rms = float(np.sqrt(np.mean(y**2))) if y.size > 0 else 0.0
        if not (RMS_MIN <= rms <= RMS_MAX):
            return []
        zcr = zero_crossing_rate(y)
        if zcr > ZCR_MAX:
            return []
        f = compute_freq_features(y, sr)
        if f["spec_mean"] < SPEC_MEAN_MIN or f["spec_std"] < SPEC_STD_MIN:
            return []


    if config.get("use_amplitude_normalization", False):
        # 先限峰再做 RMS 归一化（在 rms_normalize 内部完成）
        y, normed_rms = rms_normalize(y, target_rms=TARGET_RMS)
    else:
        normed_rms = float(np.sqrt(np.mean(y**2))) if y.size > 0 else 0.0


    duration = len(y) / sr if sr else 0.0
    if duration <= 0:
        return []


    peak = float(np.max(np.abs(y))) if y.size > 0 else 0.0
    zcr = zero_crossing_rate(y)
    freq_feats = compute_freq_features(y, sr)


    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{stem}.wav")
    ensure_dir(out_dir)
    try:
        sf.write(out_path, y, sr)
    except Exception as e:
        print(f"[ERROR] write wav failed: {out_path}, err={e}")
        return []

    stats = [{
        "file": os.path.basename(out_path),
        "orig_path": os.path.abspath(path),
        "out_path": os.path.abspath(out_path),
        "orig_duration": orig_len,
        "duration": duration,
        "rms": normed_rms,
        "zcr": zcr,
        "peak": peak,
        **freq_feats
    }]
    return stats



def batch_process_with_config(in_dir: str,
                              out_dir_base: str,
                              log_dir_base: str,
                              config_name: str,
                              config: Dict,
                              recursive: bool = False) -> str:


    out_dir = os.path.join(out_dir_base, config_name)
    log_path = os.path.join(log_dir_base, f"log_{config_name}.csv")
    print(f"\n--- Running Experiment: {config_name} ---")
    print(f"Config: {config}")
    ensure_dir(out_dir)
    ensure_dir(os.path.dirname(log_path))

    files = list_audio_files(in_dir, exts=EXTS, recursive=recursive)
    if not files:
        print(f"[WARN] No audio files found in: {in_dir}")
        return log_path

    all_stats: List[Dict] = []
    for p in tqdm(files, desc=config_name):
        stats = process_audio_for_ablation(p, out_dir, config)
        if stats:
            all_stats.extend(stats)


    if all_stats:
        keys = set()
        for r in all_stats:
            keys.update(r.keys())
        header = [
            "file", "orig_path", "out_path",
            "orig_duration", "duration", "rms", "zcr", "peak",
            "spec_mean", "spec_std", "mid_mean", "hi_mean", "low_mean"
        ]

        header = header + [k for k in sorted(keys) if k not in header]
        try:
            with open(log_path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                w.writeheader()
                for r in all_stats:
                    w.writerow({k: r.get(k, "") for k in header})
            print(f"Experiment '{config_name}' finished. Output in '{out_dir}' | Log: {log_path} | Kept: {len(all_stats)}")
        except Exception as e:
            print(f"[ERROR] write log failed: {log_path}, err={e}")
    else:
        print(f"[INFO] No files kept for '{config_name}'.")
    return log_path



def get_experiment_configs(which: str = "all") -> Dict[str, Dict]:

    configs = {
        # 仅格式统一（重采样/单声道），不做任何处理
        "baseline": {},

        # 全流程：裁头尾 + 压内部静音 + 过滤 + 幅度归一化
        "full_method": {
            "use_trim_silence": True,
            "use_compress_internal_silence": True,
            "use_feature_filtering": True,
            "use_amplitude_normalization": True,
        },

        # 关闭特征过滤
        "ablation_no_filtering": {
            "use_trim_silence": True,
            "use_compress_internal_silence": True,
            "use_feature_filtering": False,
            "use_amplitude_normalization": True,
        },

        # 关闭幅度归一化
        "ablation_no_normalization": {
            "use_trim_silence": True,
            "use_compress_internal_silence": True,
            "use_feature_filtering": True,
            "use_amplitude_normalization": False,
        },

        # 仅裁头尾，不压内部静音
        "ablation_head_tail_trim_only": {
            "use_trim_silence": True,
            "use_compress_internal_silence": False,
            "use_feature_filtering": True,
            "use_amplitude_normalization": True,
        },

        # 完全不做静音处理（既不裁头尾，也不压内部），其余保留
        "ablation_no_silence_handling": {
            "use_trim_silence": False,
            "use_compress_internal_silence": False,
            "use_feature_filtering": True,
            "use_amplitude_normalization": True,
        },

        # 不压内部静音 + 不做过滤，保留裁头尾与归一化
        "trim+norm_": {# (new_optimal_pipeline)
            "use_trim_silence": True,
            "use_compress_internal_silence": False,
            "use_feature_filtering": False,
            "use_amplitude_normalization": True,
        },
    }

    if which == "all":
        return configs
    if which == "triad":
        sel = ["ablation_no_silence_handling", "full_method", "new_optimal_pipeline"]
        return {k: configs[k] for k in sel}
    if which == "baseline":
        return {"baseline": configs["baseline"]}
    if which == "new_optimal":
        return {"new_optimal_pipeline": configs["new_optimal_pipeline"]}
    # 自定义逗号分隔
    keys = [k.strip() for k in which.split(",") if k.strip()]
    return {k: configs[k] for k in keys if k in configs}



def summarize_group_logs(log_paths: List[str], out_csv: str):

    rows = []
    for lp in log_paths:
        group = os.path.splitext(os.path.basename(lp))[0].replace("log_", "")
        if not os.path.isfile(lp):
            rows.append({"group": group, "count": 0})
            continue
        try:
            with open(lp, "r", encoding="utf-8-sig") as f:
                rd = csv.DictReader(f)
                vals = list(rd)
        except Exception:
            vals = []
        n = len(vals)
        def to_floats(col):
            out = []
            for r in vals:
                try:
                    out.append(float(r.get(col, "")))
                except Exception:
                    pass
            return np.array(out, dtype=np.float64) if out else np.array([], dtype=np.float64)
        metrics = {}
        for c in ["duration","rms","zcr","peak","spec_mean","spec_std","mid_mean","hi_mean","low_mean"]:
            v = to_floats(c)
            if v.size:
                metrics[f"mean_{c}"] = float(np.mean(v))
                metrics[f"median_{c}"] = float(np.median(v))
                metrics[f"std_{c}"] = float(np.std(v, ddof=1)) if v.size > 1 else float("nan")
        rows.append({"group": group, "count": n, **metrics})

    ensure_dir(os.path.dirname(out_csv))
    header = sorted({k for r in rows for k in r.keys()}, key=lambda x: (x!="group", x!="count", x))
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"Summary written: {out_csv}")


def write_manifest_from_group(out_dir_base: str, group: str, manifest_path: str):

    group_dir = os.path.join(out_dir_base, group)
    wavs = list_audio_files(group_dir, exts=[".wav"], recursive=True)
    ensure_dir(os.path.dirname(manifest_path))
    with open(manifest_path, "w", encoding="utf-8") as f:
        for wv in wavs:
            f.write(os.path.abspath(wv) + "\n")
    print(f"Manifest for '{group}' written: {manifest_path} (n={len(wavs)})")



def parse_args():
    ap = argparse.ArgumentParser(description="统一版训练音频预处理（消融实验）批处理器")
    ap.add_argument("--in_dir", type=str, required=True, help="输入原始音频目录")
    ap.add_argument("--out_dir_base", type=str, required=True, help="输出根目录（每组一个子目录）")
    ap.add_argument("--log_dir_base", type=str, required=True, help="日志根目录（每组一个 CSV）")
    ap.add_argument("--experiments", type=str, default="all",
                    help="运行哪些实验：all/triad/baseline/new_optimal 或 逗号分隔的键，如 full_method,ablation_no_filtering")
    ap.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    ap.add_argument("--make_manifest_for", type=str, default="",
                    help="可选：为某一组生成 manifest（组名），例如 full_method")
    ap.add_argument("--manifest_out", type=str, default="",
                    help="manifest 输出路径（txt）。与 --make_manifest_for 联用")
    return ap.parse_args()


def main():
    args = parse_args()

    configs = get_experiment_configs(args.experiments)
    if not configs:
        print(f"[ERROR] No experiments matched: {args.experiments}")
        sys.exit(1)

    ensure_dir(args.out_dir_base)
    ensure_dir(args.log_dir_base)

    log_paths = []
    for name, conf in configs.items():
        lp = batch_process_with_config(
            in_dir=args.in_dir,
            out_dir_base=args.out_dir_base,
            log_dir_base=args.log_dir_base,
            config_name=name,
            config=conf,
            recursive=args.recursive
        )
        log_paths.append(lp)


    summary_csv = os.path.join(args.log_dir_base, "ablation_summary.csv")
    summarize_group_logs(log_paths, summary_csv)


    if args.make_manifest_for and args.manifest_out:
        write_manifest_from_group(args.out_dir_base, args.make_manifest_for, args.manifest_out)

    print("\n All experiments complete.")


if __name__ == "__main__":
    main()
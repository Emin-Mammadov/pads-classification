import joblib, numpy as np
from pathlib import Path
from ..config import cfg

# ------------- BOSS settings -------------
WINDOW_SIZES = (20, 40, 80)
WINDOW_STEP  = 2
WORD_SIZE    = 4
N_BINS       = 4

# ------------- channel slices -------------
SLICES = {
    "all": slice(None),     # all 132 channels
    "acc": slice(0, 66),    # accelerometer
    "rot": slice(66, 132),  # gyroscope
}

def _load_bin(pid: str) -> np.ndarray:
    raw = np.fromfile(cfg.bins_dir / f"{pid}_ml.bin", dtype=np.float32)
    return raw.reshape(132, -1)   # (n_channels, T)

def extract_boss_features():
    """
    For each participant, writes {id}_{acc|rot|all}.joblib under cfg.feat_dir
    """
    from ml.multi_boss import MultiBOSS  # keep original dependency
    outdir = cfg.feat_dir
    bufdir = outdir / "buf"
    outdir.mkdir(parents=True, exist_ok=True)
    bufdir.mkdir(parents=True, exist_ok=True)

    pids = sorted(Path(cfg.bins_dir).glob("*.bin"))
    pids = [p.stem.split("_")[0] for p in pids]

    boss_engines = {tag: None for tag in SLICES}
    total = len(pids)

    for idx, pid in enumerate(pids, 1):
        ts = _load_bin(pid)     # (132, T)
        T  = ts.shape[1]

        for tag, sl in SLICES.items():
            out_fp = outdir / f"{pid}_{tag}.joblib"
            if out_fp.exists():
                continue

            data = ts[sl, :]  # (n_chan, T)
            n_chan = data.shape[0]

            # instantiate & fit on first encounter per tag
            if boss_engines[tag] is None:
                boss_engines[tag] = MultiBOSS(
                    data_shape=(n_chan, T),
                    window_sizes=WINDOW_SIZES,
                    window_step=WINDOW_STEP,
                    word_size=WORD_SIZE,
                    n_bins=N_BINS,
                    buf_path=str(bufdir) + "/"
                )
                X_fit = np.ascontiguousarray(data[np.newaxis, :, :], dtype=np.float64)
                boss_engines[tag].fit(X_fit, y=None)

            X_in = np.ascontiguousarray(data[np.newaxis, :, :], dtype=np.float64)
            vec  = boss_engines[tag].transform(X_in).squeeze().astype(np.float32)
            joblib.dump(vec, out_fp)

        if idx % 25 == 0 or idx == total:
            print(f"{idx}/{total} participants processed")

    n_files = len(list(outdir.glob("*.joblib")))
    print("Done –", n_files, "feature files written to", outdir)

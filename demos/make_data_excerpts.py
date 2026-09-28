"""Build the EEG excerpts in data/ from the original recordings (run once, where the recordings are).

The figures and the band-power table read only the excerpts, so the repository is self-contained. Each excerpt holds one
channel and the time span that the paper shows, with no subject identifier, date or source file name.

The locations of the original recordings are read from data/sources.local.json, which is not committed:
    {"seizure_1": "/path/to/clip.mat", "seizure_2": "/path/to/clip.mat", "sleep": "/path/to/recording_std.h5"}

Run: python demos/make_data_excerpts.py
"""
import sys, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demos"))
import numpy as np
from make_figures import _load_elrond, ictal_channel, _pretty_channel

DATA = ROOT / "data"
SEIZURES = {"seizure_1": [(292.0, 366.0)], "seizure_2": [(49.0, 183.0), (298.0, 441.0)]}     # onset and offset in s, marked by M.B.W.
SLEEP = dict(channel="c4-m1", t0=2788.0, dur=40.0, pad=5.0)                                  # 40 s of stage N2, with 5 s on each side


def main():
    src = json.loads((DATA / "sources.local.json").read_text())
    for key, ivals in SEIZURES.items():
        X, fs, ch = _load_elrond(src[key])
        X = X - X.mean(axis=0, keepdims=True)                                                  # common-average reference
        ic, rise = ictal_channel(X, fs, *ivals[0])
        out = DATA / f"eeg_{key}.npz"
        np.savez_compressed(out, x=X[ic].astype(np.float32), fs=np.float64(fs), channel=np.str_(_pretty_channel(ch[ic])),
                            reference=np.str_(f"common average of {X.shape[0]} scalp channels"), seizures_s=np.asarray(ivals, float),
                            ictal_rise_db=np.float64(rise[ic]), units=np.str_("microvolts"))
        print(f"{out.name}: channel {_pretty_channel(ch[ic])}, {X.shape[1] / fs:.0f} s at {fs:.0f} Hz, ictal rise {rise[ic]:.1f} dB, {out.stat().st_size / 1e3:.0f} kB")
    import h5py
    with h5py.File(src["sleep"], "r") as f:
        fs = int(f.attrs["sampling_rate"]); a = int((SLEEP["t0"] - SLEEP["pad"]) * fs); b = int((SLEEP["t0"] + SLEEP["dur"] + SLEEP["pad"]) * fs)
        x = f["signals/" + SLEEP["channel"]][a:b, 0].astype(float) * 1e6
        st = f["annotations_source/original/stage"][a:b, 0]
    name = SLEEP["channel"].split("-"); chn = _pretty_channel(name[0]) + "-" + name[1].upper()
    out = DATA / "eeg_sleep_n2.npz"
    np.savez_compressed(out, x=x.astype(np.float32), fs=np.float64(fs), channel=np.str_(chn), stage=st.astype(np.int8), pad_s=np.float64(SLEEP["pad"]),
                        duration_s=np.float64(SLEEP["dur"]), stage_code=np.str_("5 wake, 4 REM, 3 N1, 2 N2, 1 N3"), units=np.str_("microvolts"))
    print(f"{out.name}: channel {chn}, {len(x) / fs:.0f} s at {fs} Hz, stages {sorted(set(st.astype(int)))}, {out.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()

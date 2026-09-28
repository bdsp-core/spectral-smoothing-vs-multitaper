"""The EEG excerpts in data/: present, readable without pickle, of the stated size, and free of identifiers."""
import pathlib, re
import numpy as np

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
FILES = {"eeg_seizure_1.npz": (600.0, "C4"), "eeg_seizure_2.npz": (600.0, "Fp1"), "eeg_sleep_n2.npz": (50.0, "C4-M1")}


def test_excerpts_load_and_have_the_stated_size():
    for name, (seconds, channel) in FILES.items():
        with np.load(DATA / name, allow_pickle=False) as d:
            fs = float(d["fs"]); x = d["x"]
            assert fs == 200.0 and x.ndim == 1 and x.dtype == np.float32
            assert abs(len(x) / fs - seconds) < 0.01
            assert str(d["channel"]) == channel and str(d["units"]) == "microvolts"
            assert np.isfinite(x).all() and 1.0 < np.std(x) < 500.0              # microvolts, not volts
            if "seizures_s" in d.files:
                s = d["seizures_s"]
                assert s.ndim == 2 and s.shape[1] == 2 and (s[:, 0] < s[:, 1]).all() and s.max() < seconds
            if "stage" in d.files:
                assert len(d["stage"]) == len(x) and set(np.unique(d["stage"])) == {2}


def test_excerpts_carry_no_identifiers():
    """No subject code, date or source path in a file name, a field name or a text field."""
    bad = re.compile(r"sub-|ses-|\d{6,}|/Users/|\.mat|\.h5|\.edf", re.I)
    tracked = [p.name for p in DATA.iterdir() if p.name != "sources.local.json"]
    assert sorted(tracked) == sorted(list(FILES) + ["README.md"])
    for name in FILES:
        assert not bad.search(name)
        with np.load(DATA / name, allow_pickle=False) as d:
            for k in d.files:
                assert not bad.search(k)
                if d[k].dtype.kind in "US":
                    assert not bad.search(str(d[k]))


def test_figure_code_names_no_source_recording():
    root = DATA.parent
    for script in ("demos/make_figures.py", "demos/band_power_table.py", "demos/make_data_excerpts.py"):
        text = (root / script).read_text()
        assert not re.search(r"sub-[A-Z]\d{6,}", text), script

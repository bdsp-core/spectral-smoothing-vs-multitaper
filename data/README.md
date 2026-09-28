# EEG excerpts used in the figures

These files hold the EEG that the paper shows, and nothing more. Each holds one channel of one recording.
They carry no subject identifier, date or source file name. The recordings are de-identified clinical
recordings from the Brain Data Science Platform (BDSP); the complete recordings are available from BDSP to
credentialed users under its data-use agreement. The excerpts are released with the rest of this repository
under its license (`LICENSE.txt`).

| File | Content | Used in |
|---|---|---|
| `eeg_seizure_1.npz` | 10 min of scalp EEG at 200 Hz, channel C4 against the common average, one focal seizure | Fig. 8, Table III |
| `eeg_seizure_2.npz` | 10 min of scalp EEG at 200 Hz, channel Fp1 against the common average, two focal seizures | Table III |
| `eeg_sleep_n2.npz` | 50 s of stage N2 sleep at 200 Hz, channel C4-M1, with five sleep spindles | Fig. 9 |

The 128 Hz epoch of Figs. 1 and 7 is `legacy_matlab/InterestingSignal.mat`. All other figures use simulated records
that the scripts generate from fixed seeds.

## Fields

Read a file with `numpy.load(path, allow_pickle=False)`, or with `load_excerpt` in `demos/make_figures.py`.

| Field | Meaning |
|---|---|
| `x` | the signal in microvolts, 32-bit |
| `fs` | sampling rate in Hz |
| `channel` | channel name |
| `units` | `microvolts` |
| `reference` | seizure clips: the reference of the channel |
| `seizures_s` | seizure clips: onset and offset of each seizure in seconds, marked by M.B.W. |
| `ictal_rise_db` | seizure clips: rise in 2 to 20 Hz power from before the seizure to during it; the channel shown is the one where it is largest |
| `stage` | sleep: sleep stage at each sample (5 wake, 4 REM, 3 N1, 2 N2, 1 N3), scored by a technologist |
| `pad_s`, `duration_s` | sleep: the figure shows `duration_s` seconds, starting `pad_s` seconds into the excerpt |

## How they were made

`demos/make_data_excerpts.py` built them from the original recordings. It reads the locations of those
recordings from `data/sources.local.json`, which is kept out of the repository.

## Ethics

Retrospective analysis of EEG data was conducted under Institutional Review Board protocols at Stanford
University (#83833), Beth Israel Deaconess Medical Center (#2016P000058) and Massachusetts General Hospital
(#2013P001024), with a waiver of consent.

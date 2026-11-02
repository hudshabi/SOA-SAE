# Data

The raw datasets are not redistributed here. Download and extract them into this folder:

| Folder | Source | Required contents |
|---|---|---|
| `data/ucihar/` | https://archive.ics.uci.edu/dataset/240 | `UCI HAR Dataset/` including `Inertial Signals/` |
| `data/wisdm/` | https://www.cis.fordham.edu/wisdm/dataset.php (`WISDM_ar_latest.tar.gz`) | `WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt` |
| `data/pamap2/` | https://archive.ics.uci.edu/dataset/231 | `PAMAP2_Dataset/Protocol/subject101.dat` … `subject109.dat` |

Then run `python prep_data.py --root data`, which writes `data/cache/{ucihar,wisdm,pamap2}.npz`.

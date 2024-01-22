## Download

We provide the pre-trained checkpoints (stage 1 and 2) and top-k files on both datasets.

[click to download](https://1drv.ms/u/s!AgLqyV5O53gxuMoaukdjb9FA46bymQ?e=XQKUbh) (hosted on OneDrive), 7.637 GiB, `sha1sum: a5e29b35ceae4fcb3782b8fd9228121090ddef64`.

file content:

```bash
Archive:  candidate_rerank_ckpts.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
        0  2024-01-22 03:57   stage1/
        0  2024-01-23 02:26   stage1/fashionIQ/
        0  2024-01-23 02:26   stage1/fashionIQ/saved_models/
1995000835  2024-01-22 03:57   stage1/fashionIQ/saved_models/blip.pt
 26371563  2024-01-22 03:57   stage1/fashionIQ/fiq_top_200_val_toptee.pt
 23952043  2024-01-22 03:57   stage1/fashionIQ/fiq_top_200_val_dress.pt
 29405291  2024-01-22 03:57   stage1/fashionIQ/fiq_top_200_val_shirt.pt
        0  2024-01-23 02:26   stage1/CIRR/
        0  2024-01-22 03:57   stage1/CIRR/saved_models/
1995000835  2024-01-22 03:57   stage1/CIRR/saved_models/blip_mean.pt
 59948455  2024-01-22 03:57   stage1/CIRR/cirr_top_200_val.pt
 56473007  2024-01-22 03:57   stage1/CIRR/cirr_top_200_test1.pt
        0  2024-01-22 04:00   stage2/
        0  2024-01-22 04:00   stage2/fashionIQ/
        0  2024-01-22 04:00   stage2/fashionIQ/saved_models/
2772082830  2024-01-22 04:00   stage2/fashionIQ/saved_models/blip.pt
        0  2024-01-22 04:00   stage2/CIRR/
        0  2024-01-22 04:00   stage2/CIRR/saved_models/
2772083982  2024-01-22 04:00   stage2/CIRR/saved_models/blip_mean.pt
---------                     -------
9730318841                     19 files
```

##

After downloading, please unzip and arrange the files such that they correspond to the file structure below.

```bash
models
├── stage1
│   ├── CIRR
│   └── fashionIQ
└── stage2
    ├── CIRR
    └── fashionIQ

```
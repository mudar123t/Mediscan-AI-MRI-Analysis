# BRISC 2025 — brain tumor segmentation + classification dataset

- **Citation:** Fateh A, Rezvani Y, Moayedi S, Rezvani S, Fateh F, Fateh M, Abolghasemi V. BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification. *Scientific Data* 13, 361 (2026). doi:10.1038/s41597-026-06753-y. arXiv:2506.14318 (v5, 2026-01-28).
- **Data:** Kaggle https://www.kaggle.com/datasets/briscdataset/brisc2025/ (doi:10.34740/kaggle/ds/7632487), CC BY 4.0, ~277 MB, v6 2025-11-04. Also listed on Mendeley Data (c7dj4pwhcv — seen in search only).
- **Verified links (2026-11-19):** https://arxiv.org/abs/2506.14318 ; https://arxiv.org/html/2506.14318v5 ; https://www.kaggle.com/api/v1/datasets/view/briscdataset/brisc2025 . (nature.com redirected to a login page.)

## Facts
- 6,000 T1-weighted (CE per abstract) 2D slices; 5,000 train / 1,000 test.
- Classes (train/test): glioma 1,147/254; meningioma 1,329/306; pituitary 1,457/300; no-tumor 1,067/140.
- Planes: axial 1,550/387, coronal 1,660/316, sagittal 1,790/297.
- Masks: pixel-wise PNG per tumor image (4,793 = 6,000 − 1,207 no-tumor), validated by physicians/radiologists.
- Source: **derived from the Nickparvar Kaggle dataset** (which aggregates Cheng/figshare, SARTAJ, Br35H). Exact and near duplicates removed before splitting.
- **No patient IDs**; authors: "complete subject-level independence cannot be guaranteed due to the source limitations."
- Baseline: Swin-based model (Swin-HAFNet in v2; v2 abstract reports weighted mean IoU 82.3%).

## Relevance to Mediscan
- Only other public 2D T1-CE set with masks **and** the same 3 tumor classes → useful as a second benchmark or extra training data.
- **Cannot serve as independent external validation**: overlaps with Figshare images through Nickparvar. If used, first remove Figshare-matching images (perceptual hash / correlation, remember the column-major `.mat` transpose).
- Interesting novelty angle: quantify the Figshare↔BRISC overlap and show how cross-dataset scores change after removal.

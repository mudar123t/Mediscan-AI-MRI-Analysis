# Ferreira et al. 2024: Multi-task vs single-task brain tumor classification

**Citation:** Ferreira GM, Botelho VR, Rodrigues ÁL, Becker CDL, Alva TAP. *Modelo multi-task para classificação e segmentação de tumores cerebrais* (Multi-task model for brain tumor classification and segmentation). Journal of Health Informatics 16 (Especial, CBIS 2024), 2024. DOI 10.59681/2175-4411.v16.iEspecial.2024.1296. BibTeX key: `ferreira2024multitask`.
**Verified link (opened 2026-11-19):** https://jhi.sbis.org.br/index.php/jhi-sbis/article/view/1296 (abstract page). The PDF download could not be parsed.

## Problem
Does a multi-task (MTL) classification + segmentation model beat a single-task (ST) classifier?

## Method
- MTL: a **shared encoder** that splits into a fully connected classification branch and a segmentation decoder.
- ST: the same classification branch without the decoder.

## Data and protocol
SARTAJ + Figshare (Cheng). According to the search snippet, only Figshare images have masks, which a snippet put at about 56.7% of the data; that figure was not confirmed on the opened page. The paper uses 5-fold CV. The opened abstract does not say whether folds are patient-disjoint.

## Results
- MTL accuracy 95.99 ± 0.70% vs ST 95.87 ± 1.01%. **This is effectively no difference.**
- MTL is more stable (smaller SD on all metrics) and has far fewer parameters ("up to 187×" fewer than compared models).
- About 3% below the best literature model.

## Limitations
Conference / special-issue paper. Patient-disjointness is unclear. The mixed dataset only partly has masks.

## What we can reuse
- **Key evidence for the D12 verdict:** a shared-encoder seg+cls design is common, and when someone runs the fair ablation, the accuracy gain is about zero.
- If we build D12 anyway, for VRAM efficiency (one encoder) and stability, we must present it as engineering, not novelty, and run the same MTL-vs-ST ablation on **patient-disjoint** folds.
- Lower variance is a legitimate secondary benefit worth measuring (report SD across our 5 folds).

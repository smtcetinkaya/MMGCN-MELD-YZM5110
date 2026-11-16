# Cross-Modal Edge Gating and Minority-Class Calibration in MMGCN for Emotion Recognition in Conversation (YZM5110 Research Project)

**Author:** Samet Çetinkaya  
**Student ID:** 26526003  
**Course:** YZM5110 – Research Project (Individual, 30%)  
**Submission Date:** November 18, 2026 (Week 7)  
**Public Repository URL:** [`https://github.com/smtcetinkaya/MMGCN-MELD-YZM5110`](https://github.com/smtcetinkaya/MMGCN-MELD-YZM5110)  
**Topic:** Multimodal Graph Neural Networks for Emotion Recognition in Conversation  
**Reproduced Paper:** Hu et al. (ACL-IJCNLP 2021), *"MMGCN: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation"* ([arXiv:2107.06779](https://arxiv.org/abs/2107.06779))  
**Official Source Repository & Pinned Commit:** [`https://github.com/hujingwen6666/MMGCN`](https://github.com/hujingwen6666/MMGCN) @ `85732d984f70c1c84dd47c81aa97f1271397b899`

---

## 📄 Main Deliverables in This Repository

| File / Directory | Description |
| :--- | :--- |
| **[`Samet_Çetinkaya_YZM5110_Research.pdf`](Samet_Çetinkaya_YZM5110_Research.pdf)** | **Final Submission Report (PDF)** formatted per Section 5 of the YZM5110 syllabus (includes 5-paper comparison matrix, MELD visual overview, architectural diagrams, full results, controls, and failure analysis). |
| **[`Samet_Çetinkaya_YZM5110_Research.md`](Samet_Çetinkaya_YZM5110_Research.md)** | Full English research report in Markdown format with embedded SVG/PNG figures and complete tables. |
| **[`MMGCN_MELD_Colab_Epoch_Logs.ipynb`](MMGCN_MELD_Colab_Epoch_Logs.ipynb)** | **Notebook 1 (Baseline Reproduction):** Leak-free reproduction of MMGCN on MELD (`SEED=27350`, 90%/10% dialogue-level train/val split, single test evaluation, PyTorch 2.x compatibility patches). |
| **[`MMGCN_MELD_Colab_Experiments.ipynb`](MMGCN_MELD_Colab_Experiments.ipynb)** | **Notebook 2 (Proposed Improvement, Controls & Robustness Suite):** Implements `GatedMM_GCN` (`B0, B1, G0, G1, G1_CW`), 10-condition modality corruption grid, post-hoc logit adjustment (2A), and class-weighted `FocalLoss` (2B). |
| **[`results/`](results/)** | Saved experimental outputs: `all_summaries.json`, `clean_and_minority_table.csv`, `robustness_comparison.csv`, `gate_activations.csv`, per-variant epoch logs (`B0`..`G1_CW/history.csv`), and all report figures (`results/figures/`). |
| **[`scripts/generate_report_assets.js`](scripts/generate_report_assets.js)** | Reproducible script that reads `results/*.csv` and `results/all_summaries.json` to generate the SVG diagrams (`fig_meld_overview.svg`, `fig_architecture_gate.svg`, `fig_gates_and_perclass.svg`). |
| **[`scripts/build_pdf_report.js`](scripts/build_pdf_report.js)** | Reproducible script that compiles the print HTML and renders `Cetinkaya_YZM5110_Research.pdf`. |
| **[`requirements.txt`](requirements.txt)** | Exact Python package dependencies for Google Colab / local execution. |

---

## 🔬 Summary of Experimental Results (MELD Dataset, Tesla T4 GPU)

### 1. Clean Test Performance, Minority-Class Recovery (`fear` & `disgust`), and Resource Overhead
| Variant | Description | Val W-F1 (%) | Test W-F1 (Std) | Test Macro-F1 (Std) | `fear` F1 (Std) | `disgust` F1 (Std) | $\tau_{\text{val}}$ | Test W-F1 (LA) | Test Macro-F1 (LA) | `fear` F1 (LA) | `disgust` F1 (LA) | Params | Train Time (s) | Peak VRAM (MiB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Published** | Hu et al. (ACL 2021) *(test-selected, `valid=0.0`)* | — | **58.65** | — | — | — | — | — | — | — | — | 2,552,217 | — | — |
| **`B0` (Run 1)** | Reproduced Baseline *(unclipped)* | 70.18 | **57.44** | **35.73** | 0.00 | 2.86 | — | — | — | — | — | 2,552,217 | 974.0 | — |
| **`B0` (Run 2)** | Reproduced Baseline *(matched grad-clip)* | 70.07 | 56.60 | 34.72 | 3.51 | 0.00 | 0.25 | 54.45 | 35.24 | 5.31 | 2.15 | 2,552,217 | 1,067.7 | 149.8 |
| **`B1`** | **Control:** Un-gated + Corruption Aug | 69.45 | 56.68 | 34.19 | 0.00 | 0.00 | 0.30 | 53.73 | **35.71** | **5.87** | 4.08 | 2,552,217 | 1,065.2 | **148.6** |
| **`G0`** | **Ablation:** Cross-Modal Gate Only | 70.38 | 56.52 | 34.13 | 0.00 | 0.00 | 0.00 | 56.52 | 34.13 | 0.00 | 0.00 | 2,785,743 | 1,655.0 | 162.1 |
| **`G1`** | **Proposed:** Directed Gate + Corruption Aug | 69.95 | **57.56** | **35.32** | 0.00 | 0.00 | 0.35 | **55.21** | 35.41 | 4.65 | 0.00 | 2,785,743 | 1,646.1 | 161.9 |
| **`G1_CW`** | **Ablation 2B:** `G1` + Class-Weighted Loss | **70.73** | 55.78 | 34.51 | **3.85** | **3.70** | 0.20 | 54.53 | 35.06 | 4.95 | **6.70** | 2,785,743 | 1,657.4 | 162.1 |

### 2. Robustness Under Audio/Visual Modality Corruption (Test Weighted-F1 %)
| Corruption Scenario | `B0` (Baseline) | `B1` (Control) | `G0` (Gate Only) | **`G1` (Proposed)** | `G1_CW` (2B) | Gain (`G1` − `B0`) | Gain (`G1` − `B1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean` | 56.60 | 56.68 | 56.52 | **57.56** | 55.78 | **+0.96** | **+0.88** |
| `audio_noise_1` ($\sigma=1.0$) | 56.43 | 56.85 | 56.31 | **57.38** | 55.79 | **+0.95** | **+0.53** |
| `audio_noise_2` ($\sigma=2.0$) | 55.98 | **56.72** | 56.62 | 56.62 | 55.31 | +0.64 | -0.10 |
| `audio_zero` ($a=\mathbf{0}$) | 55.70 | 55.94 | 56.17 | **56.70** | 55.74 | **+1.00** | **+0.76** |
| `visual_noise_1` ($\sigma=1.0$) | 56.70 | 56.82 | 56.58 | **57.61** | 55.64 | **+0.91** | **+0.79** |
| `visual_noise_2` ($\sigma=2.0$) | 56.82 | 56.76 | 56.53 | **57.72** | 55.47 | **+0.90** | **+0.96** |
| `visual_zero` ($v=\mathbf{0}$) | 56.82 | 56.32 | 56.12 | **57.01** | 54.97 | +0.19 | **+0.69** |
| `av_noise_1` ($a,v$ $\sigma=1.0$) | 56.57 | 56.95 | 56.67 | **57.79** | 55.49 | **+1.22** | **+0.84** |
| `av_zero` ($a=\mathbf{0}, v=\mathbf{0}$) | 55.86 | 55.75 | 55.86 | **56.76** | 54.92 | **+0.90** | **+1.01** |
| `av_swap` (Misaligned $a,v$) | 55.39 | 55.62 | 55.85 | **56.23** | 55.62 | **+0.84** | **+0.61** |
| **Mean (9 Corruptions)** | **56.25** | **56.41** | **56.30** | **57.09** | **55.44** | **+0.84** | **+0.68** |

---

## ⚙️ Hardware Requirements, Data Access, and How to Run

### 1. Hardware & Environment Specifications
- **Recommended Environment:** Google Colab with **NVIDIA Tesla T4 GPU** (15 GB VRAM).
- **Actual Resource Footprint:** Peak GPU VRAM usage is only **149.8 MiB** (`B0/B1`) to **162.1 MiB** (`G0/G1/G1_CW`). Any CUDA-capable GPU with $\ge 2\text{ GB}$ VRAM is sufficient.
- **Software Tested:** Python `3.13.16`, PyTorch `2.14.1+cu130`, CUDA `13.0`, `torch-geometric==2.8.1`.

### 2. Data Access Instructions (MELD Public vs. IEMOCAP Licensed)
- **MELD (Public):** The notebooks automatically clone the official MMGCN repository at commit `85732d984f70c1c84dd47c81aa97f1271397b899` and read `MELD_features/MELD_features_raw1.pkl` (84.11 MiB). Per course guidelines, `.pkl` data files and large `.pt` checkpoints are excluded from this repository via `.gitignore`.
- **IEMOCAP (Restricted License — Not Used):** Access to IEMOCAP requires an institutional license from the University of Southern California (`https://sail.usc.edu/iemocap/`). Because a USC license was not available, IEMOCAP is excluded from our experimental scope.

### 3. Running the Colab Notebooks (Smoke Test vs. Full Run)
1. Upload **[`MMGCN_MELD_Colab_Experiments.ipynb`](MMGCN_MELD_Colab_Experiments.ipynb)** to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime $\to$ Change runtime type $\to$ Hardware accelerator: T4 GPU**.
3. **For a 45-second Smoke Test (Quick Verification):**
   - In Section 3 (`c04_config`), set `SMOKE_TEST = True` and run all cells (`Runtime -> Run all`). This executes 2 epochs × 2 batches per variant to verify the entire pipeline in under 1 minute.
4. **For the Full 60-Epoch Reproduction & Improvement Suite (~118 min on T4):**
   - In Section 3 (`c04_config`), set `SMOKE_TEST = False` and `VARIANTS_TO_RUN = ['B0', 'B1', 'G0', 'G1', 'G1_CW']`, then run all cells.
   - All CSV tables, JSON summaries, and PNG figures are saved to `/content/mmgcn_meld_experiments` and packaged into `MMGCN_MELD_All_Experiments_Report_Data.zip`.

### 4. Regenerating Report Figures and PDF Locally
```bash
node scripts/generate_report_assets.js
node scripts/build_pdf_report.js
```

---

## 🤖 AI-Use Disclosure (YZM5110 §7)
Disclosed AI assistance (*Antigravity AI Assistant*) was used for legacy PyTorch 1.1 compatibility patching, implementing the `GatedMM_GCN` module and evaluation scripts, and formatting report tables and SVG diagrams. All experiments on Google Colab Tesla T4 GPU were executed by **Samet Çetinkaya (26526003)**, who verified all citations, code behavior, and empirical claims.

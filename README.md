## Hardware export (experimental)

After a successful full or qat training run, export its checkpoint. Smoke weights are for interface testing only.

```powershell
.\.venv\Scripts\python.exe -m py_compile SSANproject\Modeltrain\export_weights.py SSANproject\Modeltrain\fixed_point_reference.py SSANproject\Modeltrain\verify_hex_weights.py
.\.venv\Scripts\python.exe SSANproject\Modeltrain\export_weights.py --checkpoint runs/ca_smoke/model.pt --out runs/ca_hex_smoke --data SSANproject/Modeltrain/mitdb
.\.venv\Scripts\python.exe SSANproject\Modeltrain\verify_hex_weights.py --checkpoint runs/ca_smoke/model.pt --dir runs/ca_hex_smoke
```

Use a new output directory on every export. All eight affine layers have 8-bit Po2 weights (80 zero, bit 6 sign, bits 5:0 signed exponent -7..7), 32-bit signed Q16.16 biases, C-order flattening, one word per line. Inputs are 180 ECG and 6 RR Q16.16 words; outputs are four signed Q16.16 logits. Integer reference uses 64-bit sums, arithmetic right shifts, floor division and signed 32-bit saturation. Thresholds are Q16.16 probability margins (2 disables early exit). Confidence still uses host floating softmax; a hardware confidence controller is not implemented. Manifest includes file hashes, shapes, scaling, expected class/timestep and float comparison. Synthetic and optional development-record vectors are not a clinical accuracy test. Integer replay is not independent RTL verification. Float mismatch is possible and must be investigated. No new bitstream or RTL integration is provided; do not use these files with the old diagnostic RTL. Export and verification have not yet been executed in the authoring environment.


# Enhanced CA-SSAN experimental pipeline

The original importantonlymit.py remains unchanged. New files are under SSANproject/Modeltrain.
From the repository root in Windows PowerShell, using the existing Python 3.13 virtual environment:

```powershell
git switch ca-ssan-improved
git pull origin ca-ssan-improved
.\.venv\Scripts\python.exe -m py_compile SSANproject\Modeltrain\ca_ssan_model.py SSANproject\Modeltrain\train_ca_ssan.py
.\.venv\Scripts\python.exe SSANproject\Modeltrain\train_ca_ssan.py --data SSANproject/Modeltrain/mitdb --epochs 2 --limit 512 --variant full --out runs/ca_smoke
```

Use a fresh output directory for each run. Metrics are fractions (0–1), not percentages.
Smoke results are not publication results. For full development runs remove --limit, set --epochs 80, and use distinct --out directories. Variants: reference (new architecture control, not original SSAN), balanced (focal + sampler), compact (adds projection compression), qat (adds Po2 and shift-compatible decay), full (adds contrastive, activity loss and calibrated actual early exit). These are cumulative bundles, not isolated component ablations. Use identical seeds and splits. The original SSAN needs a matched-split comparison before claiming improvement.

Record 202 is moved into development with 201, which is treated as the same patient. Validation is grouped within development; test is not loaded unless --test is explicitly requested after model selection. This is a modified DS1/DS2 protocol, not directly comparable with canonical splits. Prior test exposure cannot be undone. The inherited preprocessing uses annotated beat centers and a 250 ms post-peak window; it is not an end-to-end causal detector. Audit RR annotation handling and patient identities before publication.

No knowledge distillation, fixed-point RTL, weight export or measured hardware energy is implemented in this revision. QAT is floating-point simulation; softmax is used for exit confidence only. Existing Tcl/RTL remain diagnostic. No runtime tests have been executed on the new code by the authoring environment.

---

# SSAN / proposed CA-SSAN — Windows PC and PYNQ-Z2 testing

**Research development branch, not a validated clinical or FPGA neural accelerator.** The Python SSAN has been corrected; the current RTL is only an RR-threshold diagnostic. The proposed CA-SSAN focal loss, compression, distillation and three-stage controller are not yet implemented. Do not use historical accuracy, energy or power claims as new results.

## 1. Download the correct branch

Open PowerShell:
```powershell
git clone -b ca-ssan-improved https://github.com/chandradurga14031996-cloud/dlt.git
cd dlt
```
Alternatively download the branch ZIP from GitHub (Code > Download ZIP), extract and open PowerShell in the extracted folder.

## 2. Test Python on Windows (no FPGA required)

Install Python 3.11 (64-bit) and Git. From the repository root:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch numpy scipy scikit-learn wfdb matplotlib seaborn
.\.venv\Scripts\python.exe -m py_compile SSANproject\Modeltrain\importantonlymit.py
```
Download MIT-BIH from https://physionet.org/content/mitdb/1.0.0/ . Extract the .hea, .dat and .atr files directly into SSANproject/Modeltrain/mitdb. Verify 100.hea, 100.dat and 100.atr are present.
```powershell
cd SSANproject\Modeltrain
..\..\.venv\Scripts\python.exe -c "import torch; from importantonlymit import Proposed_SSAN_Model; m=Proposed_SSAN_Model().eval(); print(m(torch.randn(2,1,180),torch.ones(2,6)))"
..\..\.venv\Scripts\python.exe -c "from importantonlymit import run_real_mitbih_proposed_benchmark; run_real_mitbih_proposed_benchmark('mitdb')"
```
The first command is a random-input smoke test, not an accuracy test. The second runs both 80-epoch protocols and may take hours on CPU. Check printed cohort counts and validation/test separation; results include accuracy, macro F1, per-class metrics and PNG figures. Retrain from scratch. The centered beat window requires 90 post-peak samples. DS2 was previously inspected; independent external evaluation is still needed. Timestep reduction is logical, not measured energy. AUC scores for early exits use constructed logits and are not calibrated.

## 3. Vivado 2018.2 — build the legacy diagnostic only

Install Vivado 2018.2 with Zynq-7000 device support and the PYNQ-Z2 board files from https://www.tulembedded.com/FPGA/ProductsPYNQ-Z2.html . Follow the board vendor installation instructions and restart Vivado. Open Vivado with no project open. In the Tcl Console:
```tcl
source {C:/path/to/dlt/SSANproject/vivadofile/create_ssan_block_design.tcl}
```
Replace the path with your actual extracted folder. The script creates a new build_legacy_diagnostic project beside itself, applies the board preset, connects PS/DMA/reset/RTL, validates the block design and creates a wrapper. It refuses to overwrite an existing build directory. Rename old build directories after backing them up.

If the script fails, copy the first ERROR line; do not proceed with an invalid design. IP configuration and board automation have not been verified in Vivado 2018.2.

In Flow Navigator run Synthesis, then Implementation. Review utilization, timing and DRC; only generate a bitstream if the design passes. From the Tcl Console after implementation:
```tcl
open_run impl_1
report_utilization
report_timing_summary
report_power
```
Vivado report_power is an estimate, not measured board power. Use activity-based estimation and external measurement for research. Zero DSP for this heuristic does not prove Zero DSP for trained SSAN.

## 4. PYNQ-Z2 diagnostic

The existing bit/hwh are old and unchanged. Do not pair unrelated metadata. After a successful rebuild, obtain the matching generated wrapper .bit and block-design .hwh, copy them together with identical base names to a new folder on the board. The legacy input is 186 32-bit words, each carrying one signed Q4.4 value in the low byte; output is one 32-bit word. The existing tester may use synthetic data or CPU fallback: neither establishes clinical accuracy. Inspect its reported execution mode. Do not deploy this as trained neural inference.

## 5. Next work

See [SSAN_IMPROVEMENTS_STATUS.md](SSAN_IMPROVEMENTS_STATUS.md). Full neural RTL, trained weight export, bit-accurate comparison, new CA-SSAN training experiments and measured energy remain outstanding. No full training, Vivado build or board test has been performed on this branch.

## Historical README

The previous README is retained below for provenance only. Its historical claims are not verified results for this branch.

<details><summary>Original project README</summary>

# Neuromorphic SSAN Accelerator for Real-Time Cardiac Arrhythmia Classification

An end-to-end deep learning and FPGA hardware acceleration pipeline for automated, energy-efficient cardiac arrhythmia heartbeat classification based on the **MIT-BIH Arrhythmia Database**. This project integrates model training with PyTorch, Verilog-based hardware accelerator design for AMD-Xilinx Zynq-7000 SoC, PYNQ-Z2 board deployment via AXI DMA, and an interactive Hardware Diagnostic Studio UI.

---

## 📊 Dataset: MIT-BIH Arrhythmia Database

This project is trained, evaluated, and benchmarked using the gold-standard **MIT-BIH Arrhythmia Database**:

* **Database Name:** MIT-BIH Arrhythmia Database (mitdb)
* **PhysioNet Link:** [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)
* **DOI:** [https://doi.org/10.13026/C2F305](https://doi.org/10.13026/C2F305)

### Dataset Overview
The MIT-BIH Arrhythmia Database contains 48 half-hour excerpts of two-channel ambulatory ECG recordings obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979. Twenty-three recordings were chosen at random from a set of 4,000 24-hour ambulatory ECG recordings, and twenty-five recordings were selected from the same set to include less common but clinically significant arrhythmias.

* **Sampling Rate:** 360 Hz per channel with 11-bit resolution over a 10 mV range.
* **Leads:** Typically modified limb lead II (MLII) and modified lead V1 (occasionally V2, V4, or V5).
* **Target Classes (AAMI EC57 Standard Mapping):**
  1. **N (Normal):** Normal beat, Left/Right bundle branch block, Atrial escape, Nodal escape.
  2. **S (Supraventricular Ectopic Beat / SVEB):** Atrial premature beat, Aberrated atrial premature beat, Nodal premature beat, Supraventricular premature beat.
  3. **V (Ventricular Ectopic Beat / VEB):** Premature ventricular contraction, Ventricular escape beat.
  4. **F (Fusion):** Fusion of ventricular and normal beat.
  5. **Q (Unknown / Unclassifiable):** Paced beat, Unclassifiable beat.

### Citations
If utilizing this dataset or project, please reference:
1. **Moody GB, Mark RG.** *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
2. **Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE.** *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.* Circulation 101(23):e215-e220 (2000).

### Setting Up the Dataset Locally
You can download the dataset directly from PhysioNet using the `wfdb` Python package:
```bash
pip install wfdb
```
Download via Python:
```python
import wfdb
wfdb.dl_database('mitdb', dl_dir='./mitdb')
```
Or place the extracted dataset in `mit-bih-arrhythmia-database-1.0.0` or `./mitdb`.

---

## 📁 Repository Structure

```
.
├── Documents/
│   ├── Basepaer/
│   │   └── sensors-26-03723-v2 (1).pdf       # Foundational research paper
│   ├── images/
│   │   ├── Screenshot (1528).png             # Performance curves / UI screenshots
│   │   └── WhatsApp Image ...jpeg            # Hardware setup & board verification photos
│   ├── S.chandradurga.pptx                   # Project presentation slides
│   ├── S.chandradurga_updateddone.docx       # Project documentation and report (Word)
│   └── S.chandradurga_updateddone.pdf        # Project documentation and report (PDF)
│
├── SSANproject/
│   ├── Modeltrain/
│   │   └── importantonlymit.py               # 80-Epoch training pipeline on MIT-BIH with DTC
│   │
│   ├── Pynqboardfiles/
│   │   ├── gan_synthetic_test_dataset.npz    # Hardware evaluation test dataset packets
│   │   ├── ssan_pynqz2_top.bit               # Compiled FPGA bitstream for PYNQ-Z2
│   │   ├── ssan_pynqz2_top.hwh               # Hardware handoff specification for PYNQ driver
│   │   └── test_pynqz2_board.py              # On-board execution script (Jupyter/SSH)
│   │
│   ├── UI/
│   │   └── pynq_cardiac_classifier.py        # Hardware Diagnostic Studio GUI (Tkinter)
│   │
│   └── vivadofile/
│       ├── create_ssan_block_design.tcl      # Vivado IP Integrator Block Design automation
│       ├── model_classical_ssan_weights.hex  # Quantized weights hex table for BRAM/ROM
│       ├── ssan_accelerator_top.v            # Top-level Verilog hardware accelerator core
│       └── ssan_weights_rom.v                # Weight ROM module with AXI memory mapping
│
├── .gitignore                                # Git ignore rules for builds, pycache, and logs
└── README.md                                 # Complete project documentation
```

---

## ⚡ Features & Methodologies

1. **Patient-Relative RR Normalization:**
   * Uses $RR / RR_{\text{median}}$ across local sliding windows to ensure invariant cardiac rhythm representations across varied subject resting heart rates.
2. **Baseline Drift Removal:**
   * Employs high-speed cumulative sum (`cumsum`) moving average filtering equivalent to a 0.5 Hz high-pass filter, suppressing respiratory motion artifacts without phase distortion.
3. **Dynamic Timestep Controller (DTC):**
   * Early-exit neuromorphic inference enabled at timestep $t=3$ for high-confidence predictions, drastically cutting down dynamic power dissipation on edge devices.
4. **FPGA Hardware Implementation (PYNQ-Z2):**
   * Targeted for AMD-Xilinx Zynq-7020 SoC running at 100 MHz.
   * High-throughput AXI4-Stream DMA transfer directly between ARM Cortex-A9 Processing System (PS) and FPGA Programmable Logic (PL).

---

## 🚀 Quick Start Guide

### 1. Requirements & Prerequisites
* **Python:** 3.8+
* **Dependencies:**
  ```bash
  pip install torch torchvision numpy scipy matplotlib seaborn scikit-learn wfdb websocket-client
  ```
* **Hardware:**
  * AMD-Xilinx PYNQ-Z2 development board.
  * Micro-USB cable (for serial/power) and Ethernet cable (or Wi-Fi dongle).
* **EDA Tools (Optional, for re-synthesis):**
  * AMD-Xilinx Vivado ML Edition (2020.2 or later).

### 2. Training the Neural Network
To train the SSAN model on the MIT-BIH dataset:
```bash
cd SSANproject/Modeltrain
python importantonlymit.py
```
This script performs 80 epochs of training with cosine annealing learning rate scheduling, computes Protocol-A & Protocol-B cross-validation metrics, and outputs confusion matrices and radar charts.

### 3. Deploying to the PYNQ-Z2 Board
1. Transfer the contents of `SSANproject/Pynqboardfiles/` to your PYNQ board:
   ```bash
   scp SSANproject/Pynqboardfiles/* xilinx@192.168.2.99:/home/xilinx/jupyter_notebooks/ssan/
   ```
2. Log in to the PYNQ board or open a Jupyter Notebook terminal:
   ```bash
   ssh xilinx@192.168.2.99
   cd /home/xilinx/jupyter_notebooks/ssan/
   python3 test_pynqz2_board.py
   ```

### 4. Running the Desktop Diagnostic Studio UI
Launch the interactive diagnostic suite on your host computer:
```bash
cd SSANproject/UI
python pynq_cardiac_classifier.py
```
* Connects directly to the PYNQ-Z2 board via its IP address / Jupyter session.
* Loads test packets from `gan_synthetic_test_dataset.npz` or real-time ECG beats.
* Displays single-beat inference diagnosis, dynamic probability distribution, and hardware acceleration latency.

### 5. Vivado Hardware Regeneration (Optional)
To regenerate the block design in Vivado:
1. Open AMD-Xilinx Vivado.
2. In the Tcl Console, execute:
   ```tcl
   cd {path_to_repository}/SSANproject/vivadofile
   source create_ssan_block_design.tcl
   ```

---

## 👩‍💻 Author
* **S. Chandradurga**
* Department of Electronics and Communication Engineering / Computer Science
* Email: [chandradurga.kts@gmail.com](mailto:chandradurga.kts@gmail.com)


</details>

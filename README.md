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

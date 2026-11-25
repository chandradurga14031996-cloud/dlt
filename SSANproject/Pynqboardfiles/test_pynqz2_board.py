# ==============================================================================
# PYNQ-Z2 LEGACY RR-HEURISTIC DIAGNOSTIC (NOT TRAINED SSAN)
# Target Board: AMD-Xilinx PYNQ-Z2 (Zynq-7020 SoC)
# 
# Works directly in Jupyter Notebook (Untitled.ipynb) or Linux Terminal.
# Uses files already present on your board:
#   - ssan_pynqz2_top.bit & ssan_pynqz2_top.hwh
#   - gan_synthetic_test_dataset.npz (or automatic in-memory fallback)
# ==============================================================================

import os
import sys
import time
import numpy as np

CLASS_NAMES = ["Normal (N)", "SVEB (S)", "VEB (V)", "Fusion (F)"]

print("=" * 75)
print("       PYNQ-Z2 LEGACY RR-HEURISTIC DIAGNOSTIC")
print("=" * 75)

# ------------------------------------------------------------------------------
# 1. LOAD TEST DATA (From gan_synthetic_test_dataset.npz or in-memory)
# ------------------------------------------------------------------------------
npz_filename = "gan_synthetic_test_dataset.npz"

if os.path.exists(npz_filename):
    print(f"[*] Found local dataset file: '{npz_filename}'")
    data = np.load(npz_filename)
    if "hardware_packets" in data:
        test_packets = data["hardware_packets"]
        test_labels = data["labels"]
    elif "intra_hardware_packets" in data:
        test_packets = data["intra_hardware_packets"]
        test_labels = data["intra_labels"]
    else:
        k = list(data.keys())
        test_packets = data[k[0]]
        test_labels = data[k[1]]
    print(f"[+] Loaded {len(test_labels)} heartbeats from '{npz_filename}' successfully.")

    # Fix Q4.12 to Q4.4 INT8 scaling for FPGA hardware register (s_axis_tdata[7:0])
    if np.max(np.abs(test_packets)) > 128:
        print("[+] Adapting dataset scaling: Converted Q4.12 to FPGA Q4.4 INT8 format (>> 8).")
        test_packets = np.clip(test_packets >> 8, -128, 127).astype(np.int32)
else:
    print("[*] Generating 200 in-memory test heartbeats (50 per class)...")
    np.random.seed(42)
    t = np.linspace(0, 1, 180, dtype=np.float32)
    packets_list, labels_list = [], []
    # Templates: 0=N, 1=S, 2=V, 3=F
    templates = [
        (1.5, 0.50, 0.005, 2.0, [1.00, 1.00, 1.00,  0.00, 1.00, 1.00]),
        (4.0, 0.25, 0.002, 3.0, [0.45, 0.95, 0.47, -0.50, 0.80, 0.56]),
        (0.8, 0.40, 0.012, 4.0, [1.10, 0.70, 1.57,  0.40, 0.80, 1.37]),
        (2.5, 0.35, 0.004, 2.8, [0.65, 0.95, 0.68, -0.30, 0.80, 0.72])
    ]
    for cls_idx, (f, p, w, amp, rr_base) in enumerate(templates):
        for _ in range(50):
            wf = np.sin(2 * np.pi * f * t) + amp * np.exp(-((t - p)**2) / w) + 0.04 * np.random.randn(180)
            wf_norm = (wf - np.mean(wf)) / (np.std(wf) + 1e-6)
            rr = np.array(rr_base, dtype=np.float32) + 0.02 * np.random.randn(6).astype(np.float32)
            # Scale to Q4.12 Fixed-Point INT32
            full_vector = np.concatenate([wf_norm, rr]) * 4096.0
            packets_list.append(full_vector.astype(np.int32))
            labels_list.append(cls_idx)

    perm = np.random.permutation(len(labels_list))
    test_packets = np.array(packets_list)[perm]
    test_labels = np.array(labels_list)[perm]
    print(f"[+] Created {len(test_labels)} balanced in-memory heartbeats.")

# The current RTL takes exactly 186 32-bit input words, with int8 in low byte.
# This benchmark is diagnostic, not an evaluation of a trained neural network.
test_packets = np.asarray(test_packets)
if test_packets.ndim != 2 or test_packets.shape[1] != 186:
    raise ValueError("Legacy RTL expects 186 input words (180 ECG + six RR)")
if np.max(np.abs(test_packets)) > 128:
    test_packets = np.clip(test_packets.astype(np.int64) >> 8, -128, 127)
test_packets = test_packets.astype(np.int32)
num_beats = len(test_labels)
counts = [int(np.sum(test_labels == i)) for i in range(4)]
print(f"    Distribution: Normal={counts[0]} | SVEB={counts[1]} | VEB={counts[2]} | Fusion={counts[3]}\n")

# ------------------------------------------------------------------------------
# 2. INITIALIZE PYNQ FPGA OVERLAY & AXI DMA
# ------------------------------------------------------------------------------
dma = None
overlay = None
HAS_PYNQ = False

try:
    import pynq
    from pynq import Overlay, allocate
    HAS_PYNQ = True
except ImportError:
    print("[!] 'pynq' library not detected. Running in ARM CPU Emulation Mode.\n")

if HAS_PYNQ:
    bitstream_candidates = [
        "ssan_pynqz2_top.bit",
        "ssan_pynqz2_bd_wrapper.bit",
        "design_1_wrapper.bit"
    ]
    selected_bit = None
    for cand in bitstream_candidates:
        if os.path.exists(cand):
            selected_bit = cand
            break

    if selected_bit is None:
        selected_bit = bitstream_candidates[0]

    # Ensure matching .hwh file exists
    base_name = os.path.splitext(selected_bit)[0]
    expected_hwh = base_name + ".hwh"
    if not os.path.exists(expected_hwh):
        for h in ["ssan_pynqz2_top.hwh", "ssan_pynqz2_bd.hwh", "design_1.hwh"]:
            if os.path.exists(h):
                import shutil
                try:
                    shutil.copyfile(h, expected_hwh)
                    print(f"[*] Paired hardware metadata '{h}' -> '{expected_hwh}'")
                    break
                except Exception:
                    pass

    print(f"[*] Loading FPGA overlay '{selected_bit}' into Zynq PL fabric...")
    try:
        overlay = Overlay(selected_bit)
        if hasattr(overlay, "axi_dma_0"):
            dma = overlay.axi_dma_0
            print("[+] Hardware Overlay Active: AXI DMA @ 100 MHz Connected!\n")
        else:
            print("[!] Warning: axi_dma_0 IP block not found in overlay. Running in Emulation Mode.\n")
    except Exception as err:
        print(f"[!] Overlay Load Notice ({err}). Running in Emulation Mode.\n")
        dma = None

# ------------------------------------------------------------------------------
# 3. STREAM BEATS & EXECUTE CLASSIFICATION
# ------------------------------------------------------------------------------
preds = []
latencies_ms = []
timesteps_used = []

print(f"[*] Starting streaming benchmark on {num_beats} heartbeats...")
t_start_total = time.perf_counter()

if dma is not None:
    # FPGA Hardware Execution
    in_buf = allocate(shape=(186,), dtype=np.int32)
    out_buf = allocate(shape=(1,), dtype=np.uint32)

    try:
        # DMA Warmup
        in_buf[:] = test_packets[0].astype(np.int32)
        dma.recvchannel.transfer(out_buf)
        dma.sendchannel.transfer(in_buf)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
    except Exception:
        pass

    for i in range(num_beats):
        in_buf[:] = test_packets[i].astype(np.int32)

        t0 = time.perf_counter()
        # Arm receive channel first, then send
        dma.recvchannel.transfer(out_buf)
        dma.sendchannel.transfer(in_buf)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        t1 = time.perf_counter()

        raw_val = int(out_buf[0])
        pred_class = raw_val & 0x03
        exit_t = (raw_val >> 4) & 0x0F
        if exit_t == 0:
            exit_t = 10

        preds.append(pred_class)
        timesteps_used.append(exit_t)
        latencies_ms.append((t1 - t0) * 1000.0)

        if (i + 1) % 50 == 0 or (i + 1) == num_beats:
            print(f"    Progress: [{i + 1:3d}/{num_beats}] beats | Current Avg Latency: {np.mean(latencies_ms):.3f} ms/beat")

    try:
        in_buf.freebuffer()
        out_buf.freebuffer()
    except Exception:
        pass

else:
    # CPU Software Emulation Execution
    for i in range(num_beats):
        t0 = time.perf_counter()
        # Extract RR ratio: packet index 182
        val = test_packets[i]
        rr_ratio = float(val[182]) / 4096.0 if np.max(np.abs(val)) > 128 else float(val[182]) / 16.0

        if 0.85 <= rr_ratio <= 1.15:
            pred_class = 0  # Normal
            exit_t = 4       # Legacy bitstream exits after four iterations
        elif rr_ratio <= 0.55:
            pred_class = 1  # SVEB
            exit_t = 10
        elif rr_ratio >= 1.25:
            pred_class = 2  # VEB
            exit_t = 10
        else:
            pred_class = 3  # Fusion
            exit_t = 10

        time.sleep(0.00005)
        t1 = time.perf_counter()

        preds.append(pred_class)
        timesteps_used.append(exit_t)
        latencies_ms.append((t1 - t0) * 1000.0)

        if (i + 1) % 50 == 0 or (i + 1) == num_beats:
            print(f"    Progress: [{i + 1:3d}/{num_beats}] beats | Current Avg Latency: {np.mean(latencies_ms):.3f} ms/beat")

t_end_total = time.perf_counter()
total_sec = t_end_total - t_start_total

# ------------------------------------------------------------------------------
# 4. COMPUTE METRICS, ACCURACY & CONFUSION MATRIX
# ------------------------------------------------------------------------------
preds = np.array(preds)
cm = np.zeros((4, 4), dtype=int)
for true_lbl, pred_lbl in zip(test_labels, preds):
    cm[true_lbl, pred_lbl] += 1

accuracy = (np.trace(cm) / np.sum(cm)) * 100.0
avg_lat = float(np.mean(latencies_ms))
throughput = num_beats / total_sec
avg_timesteps = float(np.mean(timesteps_used))
energy_savings = (1.0 - (avg_timesteps / 10.0)) * 100.0

mode_str = "FPGA LEGACY HEURISTIC RTL (DMA + software timing)" if (dma is not None) else "CPU HEURISTIC EMULATION"

print("\n" + "=" * 75)
print(f"  CLASSIFICATION REPORT & HARDWARE METRICS ({mode_str})")
print("=" * 75)
print(f"  - Total Beats Evaluated    : {num_beats}")
print(f"  - Overall Accuracy         : {accuracy:.2f}%")
print(f"  - Measured Avg Latency     : {avg_lat:.3f} ms / beat")
print(f"  - Measured Throughput      : {throughput:.1f} beats / second")
print(f"  - Average Timesteps (DTC)  : {avg_timesteps:.2f} / 10")
print(f"  - Theoretical Timestep Reduction  : {energy_savings:.1f}%")
print(f"  - Total Test Duration      : {total_sec:.3f} seconds")

print("\nPer-Class Sensitivity & Specificity Breakdown:")
f1_list = []
for c in range(4):
    tp = cm[c, c]
    fp = np.sum(cm[:, c]) - tp
    fn = np.sum(cm[c, :]) - tp
    precision = (tp / (tp + fp + 1e-9)) * 100.0
    recall = (tp / (tp + fn + 1e-9)) * 100.0
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    f1_list.append(f1)
    print(f"  {CLASS_NAMES[c]:14s} | Precision: {precision:6.2f}% | Recall: {recall:6.2f}% | F1: {f1:6.2f}%")

macro_f1 = float(np.mean(f1_list))
print(f"\n  Macro-Averaged F1-Score    : {macro_f1:.2f}%")

print("\n4x4 Hardware Confusion Matrix (Rows: Ground Truth, Cols: Predicted):")
header_str = "           " + " ".join([f"{name:>8s}" for name in ["Pred N", "Pred S", "Pred V", "Pred F"]])
print(header_str)
for r in range(4):
    row_label = f"True {CLASS_NAMES[r][0]}:"
    vals_str = " ".join([f"{cm[r, c]:8d}" for c in range(4)])
    print(f"  {row_label:<9s} {vals_str}")
print("=" * 75 + "\n")
print("[!] Diagnostic only: not trained SSAN inference, clinical validation, or measured energy.")
print("[+] Diagnostic test completed.")


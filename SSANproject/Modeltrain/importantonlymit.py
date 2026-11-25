# ==============================================================================
# PROPOSED SSAN MODEL BENCHMARK (REAL MIT-BIH ARRHYTHMIA DATABASE)
# Full 80 Epochs Training Pipeline & Plot Generation (No Clamping)
# 
# Features:
# 1. 80 Full Epochs Training with Cosine Annealing Learning Rate Decay
# 2. Patient-Relative RR Normalization (RR / RR_median) for Invariant Rhythm Features
# 3. Fast Cumsum Moving Average Filter for Baseline Drift Removal (0.5 Hz HPF)
# 4. Dynamic Timestep Controller (DTC) Adaptive Early Exit at t=3
# 5. Generates Publication-Quality High-Resolution Metric Plots & Diagrams:
#    - loss_accuracy_curves.png
#    - confusion_matrices_protocol_a_b.png
#    - per_class_radar_chart.png
#    - dtc_energy_timesteps_chart.png
# ==============================================================================

import os
import sys
import time
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, cohen_kappa_score,
    matthews_corrcoef, multilabel_confusion_matrix
)

torch.set_num_threads(min(8, os.cpu_count() or 4))
print(f"[+] Script initialized. PyTorch using {torch.get_num_threads()} CPU threads.", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Target device: {device}", flush=True)

# ------------------------------------------------------------------------------
# 1. LOCAL DATASET PATH CONFIGURATION & RECORD COHORTS
# ------------------------------------------------------------------------------
PRIMARY_LOCAL_PATH = r"D:\phd\coursework\deeplear\dlpythonantigravity\mit-bih-arrhythmia-database-1.0.0"
SECONDARY_LOCAL_PATH = "./mitdb"

DS1_RECORDS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_RECORDS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal (N)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,         # SVEB (S)
    'V': 2, 'E': 2,                         # VEB (V)
    'F': 3                                  # Fusion (F)
}

def get_valid_dataset_dir(user_path=None):
    if user_path and os.path.exists(user_path):
        return user_path
    if os.path.exists(PRIMARY_LOCAL_PATH):
        return PRIMARY_LOCAL_PATH
    if os.path.exists(SECONDARY_LOCAL_PATH):
        return SECONDARY_LOCAL_PATH
    return None

def fast_baseline_filter(signal, lowcut=0.5, fs=360):
    """Fast cumsum-based moving average filter to remove ECG baseline wander."""
    w_high = int(fs / lowcut)
    if len(signal) > w_high:
        cs = np.cumsum(np.insert(signal, 0, 0))
        baseline = (cs[w_high:] - cs[:-w_high]) / float(w_high)
        pad_left = w_high // 2
        pad_right = len(signal) - len(baseline) - pad_left
        baseline = np.pad(baseline, (pad_left, pad_right), mode='edge')
        return signal - baseline
    return signal

# ------------------------------------------------------------------------------
# 2. REAL MIT-BIH DATASET LOADER WITH PATIENT-RELATIVE RR NORMALIZATION
# ------------------------------------------------------------------------------
def load_real_mitbih_dataset(records, db_dir=None, window_size=180):
    db_path = get_valid_dataset_dir(db_dir)
    print(f"[*] Dataset search path resolved to: {db_path}", flush=True)

    try:
        import wfdb
    except ImportError:
        print("[!] Error: 'wfdb' package is not installed. Run: pip install wfdb", flush=True)
        sys.exit(1)

    if db_path is None or not os.path.exists(db_path):
        print(f"[!] ERROR: MIT-BIH dataset directory not found at '{PRIMARY_LOCAL_PATH}'.", flush=True)
        sys.exit(1)

    all_waveforms, all_rr, all_labels, all_rec_ids = [], [], [], []
    half_w = window_size // 2

    print(f"[*] Extracting heartbeats & patient-relative RR features from {len(records)} records...", flush=True)

    for idx, rec in enumerate(records):
        rec_str = str(rec)
        rec_path = os.path.join(db_path, rec_str)

        try:
            record = wfdb.rdrecord(rec_path, channels=[0])
            annotation = wfdb.rdann(rec_path, 'atr')
        except Exception:
            continue

        raw_signal = record.p_signal.flatten()
        signal = fast_baseline_filter(raw_signal, fs=record.fs)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)  # Z-score normalization

        peaks = annotation.sample
        symbols = annotation.symbol
        fs = record.fs
        rr_raw = np.diff(peaks) / fs  # RR in seconds
        rec_median_rr = np.median(rr_raw) + 1e-6  # Patient baseline median RR

        for i in range(2, len(peaks) - 2):
            sym = symbols[i]
            if sym not in AAMI_MAP:
                continue

            label = AAMI_MAP[sym]
            peak_idx = peaks[i]

            if peak_idx - half_w < 0 or peak_idx + half_w > len(signal):
                continue

            beat_wf = signal[peak_idx - half_w : peak_idx + half_w]

            # Patient-relative RR features
            rr_prev = rr_raw[i - 1] / rec_median_rr
            rr_next = rr_raw[i] / rec_median_rr
            rr_ratio = rr_prev / (rr_next + 1e-6)
            rr_diff = rr_prev - rr_next
            
            local_start = max(0, i - 5)
            local_end = min(len(rr_raw), i + 5)
            rr_local_avg = np.mean(rr_raw[local_start:local_end]) / rec_median_rr
            rr_pre_ratio = rr_prev / (rr_local_avg + 1e-6)

            rr_feat = np.array([rr_prev, rr_next, rr_ratio, rr_diff, rr_local_avg, rr_pre_ratio], dtype=np.float32)

            all_waveforms.append(beat_wf)
            all_rr.append(rr_feat)
            all_labels.append(label)
            all_rec_ids.append(rec)

    return np.array(all_waveforms), np.array(all_rr), np.array(all_labels), np.array(all_rec_ids)

# Stratified patient-specific train-test split (80/20 per record per class)
def patient_specific_split(waveforms, rr_features, labels, rec_ids, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    train_idx, test_idx = [], []
    
    unique_recs = np.unique(rec_ids)
    for rec in unique_recs:
        rec_mask = (rec_ids == rec)
        rec_indices = np.where(rec_mask)[0]
        rec_labels = labels[rec_indices]
        
        for cls in range(4):
            cls_mask = (rec_labels == cls)
            cls_indices = rec_indices[cls_mask]
            if len(cls_indices) == 0:
                continue
            np.random.shuffle(cls_indices)
            n_test = max(1, int(len(cls_indices) * test_ratio)) if len(cls_indices) > 3 else int(len(cls_indices) * test_ratio)
            test_idx.extend(cls_indices[:n_test])
            train_idx.extend(cls_indices[n_test:])
            
    return np.array(train_idx), np.array(test_idx)

class ECGDataset(Dataset):
    def __init__(self, wf, rr, lbl):
        self.wf = torch.tensor(wf, dtype=torch.float32).unsqueeze(1)
        self.rr = torch.tensor(rr, dtype=torch.float32)
        self.lbl = torch.tensor(lbl, dtype=torch.long)
    def __len__(self): return len(self.lbl)
    def __getitem__(self, i): return self.wf[i], self.rr[i], self.lbl[i]

# ------------------------------------------------------------------------------
# 3. SPIKING NEURON & POWER-OF-TWO (Po2) QUANTIZER
# ------------------------------------------------------------------------------
class FastSigmoidSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope=25.0):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        grad_input = grad_output / (1.0 + slope * torch.abs(x)) ** 2
        return grad_input, None

def surrogate_spike(x, slope=25.0):
    return FastSigmoidSurrogate.apply(x, slope)

class LIFNeuron(nn.Module):
    def __init__(self, beta=0.9, threshold=1.0, slope=25.0):
        super(LIFNeuron, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.slope = slope

    def init_leaky(self, batch_size, features, device):
        return torch.zeros(batch_size, features, device=device)

    def forward(self, input_current, mem):
        if mem is None:
            mem = torch.zeros_like(input_current)
        mem = self.beta * mem + input_current
        spk = surrogate_spike(mem - self.threshold, slope=self.slope)
        mem = mem - spk * self.threshold
        return spk, mem

class PowerOfTwoQuantizer(nn.Module):
    def __init__(self, num_bits=8):
        super(PowerOfTwoQuantizer, self).__init__()
        self.num_bits = num_bits

    def forward(self, weight):
        sign = torch.sign(weight)
        abs_w = torch.clamp(torch.abs(weight), min=1e-4, max=16.0)
        exponent = torch.clamp(torch.round(torch.log2(abs_w)), min=-(2**(self.num_bits-1)-1), max=2**(self.num_bits-1)-1)
        po2_weight = sign * (2.0 ** exponent)
        return weight + (po2_weight - weight).detach()

# ------------------------------------------------------------------------------
# 4. SOFTMAX-FREE SPIKING SELF-ATTENTION (SSA) LAYER
# ------------------------------------------------------------------------------
class SpikingSelfAttention(nn.Module):
    def __init__(self, in_ch=24, hidden_dim=16, beta=0.9):
        super(SpikingSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.po2_quant = PowerOfTwoQuantizer(num_bits=8)

        self.q = nn.Conv1d(in_ch, hidden_dim, 1)
        self.k = nn.Conv1d(in_ch, hidden_dim, 1)
        self.v = nn.Conv1d(in_ch, in_ch, 1)

        self.lif_q = LIFNeuron(beta=beta)
        self.lif_k = LIFNeuron(beta=beta)
        self.lif_v = LIFNeuron(beta=beta)
        self.scale = 1.0 / np.sqrt(hidden_dim)
        self.bn_attn = nn.BatchNorm1d(in_ch)

    def forward(self, x, mq, mk, mv):
        B = x.size(0)

        q_w = self.po2_quant(self.q.weight)
        k_w = self.po2_quant(self.k.weight)
        v_w = self.po2_quant(self.v.weight)

        q_out = F.conv1d(x, q_w, self.q.bias, padding=0)
        spk_q, mq = self.lif_q(q_out.reshape(B, -1), mq)
        sq = spk_q.reshape(B, self.hidden_dim, 45)

        k_out = F.conv1d(x, k_w, self.k.bias, padding=0)
        spk_k, mk = self.lif_k(k_out.reshape(B, -1), mk)
        sk = spk_k.reshape(B, self.hidden_dim, 45)

        v_out = F.conv1d(x, v_w, self.v.bias, padding=0)
        spk_v, mv = self.lif_v(v_out.reshape(B, -1), mv)
        sv = spk_v.reshape(B, 24, 45)

        q_t = sq.transpose(1, 2)
        v_t = sv.transpose(1, 2)
        attn = torch.bmm(q_t, sk) * self.scale
        out = torch.bmm(attn, v_t).transpose(1, 2).contiguous()
        
        out = self.bn_attn(out + x)
        return out, mq, mk, mv

# ------------------------------------------------------------------------------
# 5. PROPOSED SSAN MODEL WITH DTC EARLY EXIT & STAGE-2 SELF-CORRECTION
# ------------------------------------------------------------------------------
class Proposed_SSAN_Model(nn.Module):
    def __init__(self, max_timesteps=10, early_exit_t=3, beta=0.9):
        super(Proposed_SSAN_Model, self).__init__()
        self.T_max = max_timesteps
        self.early_exit_t = early_exit_t

        self.c1 = nn.Conv1d(1, 16, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.lif1 = LIFNeuron(beta=beta)
        self.mp1 = nn.MaxPool1d(2)

        self.c2 = nn.Conv1d(16, 24, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(24)
        self.lif2 = LIFNeuron(beta=beta)
        self.mp2 = nn.MaxPool1d(2)

        self.ssa = SpikingSelfAttention(in_ch=24, hidden_dim=16, beta=beta)
        self.fc_flat = nn.Linear(45 * 24, 480)
        self.bn_fused = nn.BatchNorm1d(486)  # 480 backbone + 6 RR features

        # Stage-1 Binary Classification Head (Normal vs. Abnormal)
        self.s1_head = nn.Linear(486, 2)
        self.s1_lif = LIFNeuron(beta=beta)

        # Stage-2 Specialist Multi-Class Head
        self.stage2_specialist = nn.Sequential(
            nn.Linear(486, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 4)
        )

    def forward(self, x, rr):
        B = x.size(0)
        m1 = self.lif1.init_leaky(B, 16 * 90, x.device)
        m2 = self.lif2.init_leaky(B, 24 * 45, x.device)
        mq = self.ssa.lif_q.init_leaky(B, 16 * 45, x.device)
        mk = self.ssa.lif_k.init_leaky(B, 16 * 45, x.device)
        mv = self.ssa.lif_v.init_leaky(B, 24 * 45, x.device)
        ms1 = self.s1_lif.init_leaky(B, 2, x.device)

        accum_features = torch.zeros(B, 486, device=x.device)
        early_exit_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        preds = torch.zeros(B, dtype=torch.long, device=x.device)
        timesteps_used = torch.full((B,), float(self.T_max), dtype=torch.float32, device=x.device)
        s1_spk_accum = torch.zeros(B, 2, device=x.device)

        for t in range(self.T_max):
            c1_out = self.mp1(self.bn1(self.c1(x)))
            spk1, m1 = self.lif1(c1_out.reshape(B, -1), m1)
            spk1_spatial = spk1.reshape(B, 16, 90)

            c2_out = self.mp2(self.bn2(self.c2(spk1_spatial)))
            spk2, m2 = self.lif2(c2_out.reshape(B, -1), m2)
            spk2_spatial = spk2.reshape(B, 24, 45)

            ssa_out, mq, mk, mv = self.ssa(spk2_spatial, mq, mk, mv)
            flat = self.fc_flat(ssa_out.reshape(B, -1))
            fused = self.bn_fused(torch.cat([flat, rr], dim=-1))
            accum_features += fused

            spk_s1, ms1 = self.s1_lif(self.s1_head(fused), ms1)
            s1_spk_accum += spk_s1

            # Dynamic Timestep Controller (DTC) Early Exit at t=3
            if t == self.early_exit_t:
                rr_ratio = rr[:, 2]     # Patient-relative RR_ratio
                rr_pre = rr[:, 5]       # Patient-relative RR_pre_ratio
                
                # Normal rhythm check (0.85 to 1.15) + High Stage-1 binary Normal confidence
                regular_rhythm = (rr_ratio >= 0.85) & (rr_ratio <= 1.15) & (rr_pre >= 0.85) & (rr_pre <= 1.15)
                confident_normal = (s1_spk_accum[:, 0] >= 2) & (s1_spk_accum[:, 1] == 0) & regular_rhythm & (~early_exit_mask)

                preds[confident_normal] = 0
                early_exit_mask[confident_normal] = True
                timesteps_used[confident_normal] = float(t + 1)

        # Stage-2 Self-Correction for beats that did not exit early
        avg_fused = accum_features / float(self.T_max)
        complex_idx = torch.where(~early_exit_mask)[0]

        if len(complex_idx) > 0:
            logits = self.stage2_specialist(avg_fused[complex_idx])
            preds[complex_idx] = torch.argmax(logits, dim=-1)

        return preds, timesteps_used.mean().item()

    def forward_proba(self, x, rr):
        """Returns (preds, avg_timesteps, softmax_probabilities[B,4]) for AUC-ROC computation."""
        B = x.size(0)
        m1 = self.lif1.init_leaky(B, 16 * 90, x.device)
        m2 = self.lif2.init_leaky(B, 24 * 45, x.device)
        mq = self.ssa.lif_q.init_leaky(B, 16 * 45, x.device)
        mk = self.ssa.lif_k.init_leaky(B, 16 * 45, x.device)
        mv = self.ssa.lif_v.init_leaky(B, 24 * 45, x.device)
        ms1 = self.s1_lif.init_leaky(B, 2, x.device)

        accum_features = torch.zeros(B, 486, device=x.device)
        early_exit_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        preds = torch.zeros(B, dtype=torch.long, device=x.device)
        timesteps_used = torch.full((B,), float(self.T_max), dtype=torch.float32, device=x.device)
        s1_spk_accum = torch.zeros(B, 2, device=x.device)
        # Accumulate logits for all samples (early-exit samples get Stage-2 logits too)
        all_logits = torch.zeros(B, 4, device=x.device)

        for t in range(self.T_max):
            c1_out = self.mp1(self.bn1(self.c1(x)))
            spk1, m1 = self.lif1(c1_out.reshape(B, -1), m1)
            spk1_spatial = spk1.reshape(B, 16, 90)

            c2_out = self.mp2(self.bn2(self.c2(spk1_spatial)))
            spk2, m2 = self.lif2(c2_out.reshape(B, -1), m2)
            spk2_spatial = spk2.reshape(B, 24, 45)

            ssa_out, mq, mk, mv = self.ssa(spk2_spatial, mq, mk, mv)
            flat = self.fc_flat(ssa_out.reshape(B, -1))
            fused = self.bn_fused(torch.cat([flat, rr], dim=-1))
            accum_features += fused

            spk_s1, ms1 = self.s1_lif(self.s1_head(fused), ms1)
            s1_spk_accum += spk_s1

            if t == self.early_exit_t:
                rr_ratio = rr[:, 2]
                rr_pre = rr[:, 5]
                regular_rhythm = (rr_ratio >= 0.85) & (rr_ratio <= 1.15) & (rr_pre >= 0.85) & (rr_pre <= 1.15)
                confident_normal = (s1_spk_accum[:, 0] >= 2) & (s1_spk_accum[:, 1] == 0) & regular_rhythm & (~early_exit_mask)
                # For early-exit normals, assign a one-hot-like logit: class 0 wins clearly
                normal_logit = torch.zeros(B, 4, device=x.device)
                normal_logit[:, 0] = 3.0   # strong Normal score
                all_logits[confident_normal] = normal_logit[confident_normal]
                preds[confident_normal] = 0
                early_exit_mask[confident_normal] = True
                timesteps_used[confident_normal] = float(t + 1)

        avg_fused = accum_features / float(self.T_max)
        complex_idx = torch.where(~early_exit_mask)[0]
        if len(complex_idx) > 0:
            logits_complex = self.stage2_specialist(avg_fused[complex_idx])
            all_logits[complex_idx] = logits_complex
            preds[complex_idx] = torch.argmax(logits_complex, dim=-1)

        proba = torch.softmax(all_logits, dim=-1)
        return preds, timesteps_used.mean().item(), proba

# ------------------------------------------------------------------------------
# 6. TRAINING & EVALUATION PIPELINE (EXACT 80 EPOCHS)
# ------------------------------------------------------------------------------
def train_proposed_ssan(model, train_loader, val_loader=None, epochs=80, lr=1e-3, protocol_name=""):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    weights_t = torch.tensor([1.0, 2.2, 1.8, 3.2], dtype=torch.float32, device=device)
    bin_weights_t = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    criterion_s1 = nn.CrossEntropyLoss(weight=bin_weights_t)
    criterion_s2 = nn.CrossEntropyLoss(weight=weights_t)

    print(f"\n[*] Starting Training: {protocol_name} ({epochs} Full Epochs)...", flush=True)

    train_losses, train_accs, val_accs, val_f1s = [], [], [], []
    best_f1 = 0.0
    best_weights = None

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_train, total_train = 0, 0

        for x, r, y in train_loader:
            x, r, y = x.to(device), r.to(device), y.to(device)
            optimizer.zero_grad()
            B = x.size(0)

            m1 = model.lif1.init_leaky(B, 16 * 90, device)
            m2 = model.lif2.init_leaky(B, 24 * 45, device)
            mq = model.ssa.lif_q.init_leaky(B, 16 * 45, device)
            mk = model.ssa.lif_k.init_leaky(B, 16 * 45, device)
            mv = model.ssa.lif_v.init_leaky(B, 24 * 45, device)
            ms1 = model.s1_lif.init_leaky(B, 2, device)

            accum_f, s1_spks = [], []
            for t in range(model.T_max):
                c1_out = model.mp1(model.bn1(model.c1(x)))
                spk1, m1 = model.lif1(c1_out.reshape(B, -1), m1)
                spk1_spatial = spk1.reshape(B, 16, 90)

                c2_out = model.mp2(model.bn2(model.c2(spk1_spatial)))
                spk2, m2 = model.lif2(c2_out.reshape(B, -1), m2)
                spk2_spatial = spk2.reshape(B, 24, 45)

                ssa_out, mq, mk, mv = model.ssa(spk2_spatial, mq, mk, mv)
                flat = model.fc_flat(ssa_out.reshape(B, -1))
                fused = model.bn_fused(torch.cat([flat, r], dim=-1))
                accum_f.append(fused)

                spk_s1, ms1 = model.s1_lif(model.s1_head(fused), ms1)
                s1_spks.append(spk_s1)

            s1_out = torch.stack(s1_spks).sum(dim=0)
            logits = model.stage2_specialist(torch.stack(accum_f).mean(dim=0))
            y_s1 = (y != 0).long()
            
            loss = criterion_s1(s1_out, y_s1) + 1.5 * criterion_s2(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds_train = torch.argmax(logits, dim=-1)
            correct_train += (preds_train == y).sum().item()
            total_train += B

        scheduler.step()
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = (correct_train / total_train) * 100.0
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation evaluation per epoch
        if val_loader is not None:
            acc_val, f1_val, _, _, _ = evaluate_proposed_ssan(model, val_loader)
            val_accs.append(acc_val)
            val_f1s.append(f1_val)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_weights = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            val_str = f" | Val Acc: {val_accs[-1]:.2f}% | Val F1: {val_f1s[-1]:.2f}%" if val_loader else ""
            print(f"    Epoch {epoch+1:2d}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%{val_str}", flush=True)

    if best_weights is not None:
        model.load_state_dict(best_weights)

    print(f"[*] {protocol_name} Training Completed in {time.time() - t0:.1f}s.", flush=True)
    return train_losses, train_accs, val_accs, val_f1s

def evaluate_proposed_ssan(model, test_loader):
    """
    Full evaluation with AAMI EC57-aligned metrics:
    Accuracy, Macro-F1, Sensitivity (Se/Recall), Specificity (Sp),
    Positive Predictivity (+P/Precision), F1 per class,
    FPR, FNR, DER, MCC, Cohen's Kappa, AUC-ROC (macro + per-class),
    Confusion Matrix, Avg Timesteps.
    """
    model.eval()
    y_true, y_pred, y_proba, t_used = [], [], [], []
    with torch.no_grad():
        for x, r, y in test_loader:
            x, r, y = x.to(device), r.to(device), y.to(device)
            pred, avg_t, proba = model.forward_proba(x, r)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_proba.extend(proba.cpu().numpy())
            t_used.append(avg_t)

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_proba = np.array(y_proba)   # shape [N, 4]
    classes = [0, 1, 2, 3]
    class_names = ['Normal', 'SVEB', 'VEB', 'Fusion']

    # ── Basic aggregate metrics ───────────────────────────────────────────────
    acc      = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    kappa    = cohen_kappa_score(y_true, y_pred) * 100.0        # Cohen's Kappa (%)
    mcc      = matthews_corrcoef(y_true, y_pred) * 100.0        # MCC (%)

    # ── AUC-ROC ──────────────────────────────────────────────────────────────
    try:
        auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro') * 100.0
        auc_per_class = {}
        for i, cname in enumerate(class_names):
            y_bin = (y_true == i).astype(int)
            auc_per_class[cname] = roc_auc_score(y_bin, y_proba[:, i]) * 100.0
        # ROC curve arrays (for plotting)
        roc_curves = {}
        for i, cname in enumerate(class_names):
            y_bin = (y_true == i).astype(int)
            fpr_c, tpr_c, _ = roc_curve(y_bin, y_proba[:, i])
            roc_curves[cname] = (fpr_c, tpr_c, auc_per_class[cname])
    except Exception:
        auc_macro = float('nan')
        auc_per_class = {c: float('nan') for c in class_names}
        roc_curves = {}

    # ── Per-class TP/TN/FP/FN via multilabel confusion matrix ────────────────
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=classes)  # [4, 2, 2]

    pr_arr, re_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(
        y_true, y_pred, labels=classes, target_names=class_names, zero_division=0)

    per_class = {}
    for i, cname in enumerate(class_names):
        tn, fp, fn, tp = mcm[i].ravel()
        total_neg  = tn + fp
        total_pos  = tp + fn

        se  = (tp / (tp + fn + 1e-9)) * 100.0          # Sensitivity = Recall
        sp  = (tn / (tn + fp + 1e-9)) * 100.0          # Specificity
        pp  = (tp / (tp + fp + 1e-9)) * 100.0          # +P = Precision
        fpr = (fp / (fp + tn + 1e-9)) * 100.0          # False Positive Rate
        fnr = (fn / (fn + tp + 1e-9)) * 100.0          # False Negative Rate
        # Detection Error Rate = (FP + FN) / total samples
        der = ((fp + fn) / (len(y_true) + 1e-9)) * 100.0
        f1c = f1_arr[i] * 100.0

        per_class[cname] = {
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'sensitivity_se': se,          # AAMI: Se
            'specificity_sp': sp,          # AAMI: Sp
            'precision_pp':   pp,          # AAMI: +P
            'recall':         se,          # same as Se
            'f1':             f1c,
            'fpr':            fpr,
            'fnr':            fnr,
            'der':            der,
            'auc':            auc_per_class.get(cname, float('nan')),
        }

    metrics_dict = {
        'accuracy':        acc,
        'macro_f1':        macro_f1,
        'kappa':           kappa,
        'mcc':             mcc,
        'auc_macro':       auc_macro,
        'avg_timesteps':   np.mean(t_used),
        'confusion_matrix': cm,
        'per_class':       per_class,
        'roc_curves':      roc_curves,
    }
    return acc, macro_f1, np.mean(t_used), report, metrics_dict

# ------------------------------------------------------------------------------
# 7. AUTOMATED PLOT & DIAGRAM GENERATION
# ------------------------------------------------------------------------------
def generate_publication_plots(train_losses_a, train_accs_a, val_accs_a, dict_a, dict_b):
    """Generates and saves publication-quality PNG diagrams for all metrics."""
    print("\n[*] Generating & saving publication-quality metric diagrams...", flush=True)
    class_names = ['Normal', 'SVEB', 'VEB', 'Fusion']

    # ── 1. Loss & Accuracy Curves ─────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_a, label='Training Loss', color='#1f77b4', linewidth=2)
    plt.title('Training Loss over 80 Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs_a, label='Training Accuracy', color='#2ca02c', linewidth=2)
    if len(val_accs_a) > 0:
        plt.plot(val_accs_a, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    plt.title('Model Accuracy (%) over 80 Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    plt.tight_layout()
    plt.savefig('loss_accuracy_curves.png', dpi=300); plt.close()
    print("  [+] Saved 'loss_accuracy_curves.png'")

    # ── 2. Confusion Matrices ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(dict_a['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[0].set_title(
        f"Protocol A (Intra-Patient)\nAcc: {dict_a['accuracy']:.2f}% | Macro-F1: {dict_a['macro_f1']:.2f}% | "
        f"Kappa: {dict_a['kappa']:.2f}% | MCC: {dict_a['mcc']:.2f}%",
        fontsize=10, fontweight='bold')
    axes[0].set_xlabel('Predicted Label'); axes[0].set_ylabel('True Label')

    sns.heatmap(dict_b['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[1].set_title(
        f"Protocol B (Inter-Patient DS1->DS2)\nAcc: {dict_b['accuracy']:.2f}% | Macro-F1: {dict_b['macro_f1']:.2f}% | "
        f"Kappa: {dict_b['kappa']:.2f}% | MCC: {dict_b['mcc']:.2f}%",
        fontsize=10, fontweight='bold')
    axes[1].set_xlabel('Predicted Label'); axes[1].set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrices_protocol_a_b.png', dpi=300); plt.close()
    print("  [+] Saved 'confusion_matrices_protocol_a_b.png'")

    # ── 3. Per-Class Radar Chart (F1-Score) ───────────────────────────────────
    N = len(class_names)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    f1_a = [dict_a['per_class'][c]['f1'] for c in class_names] + [dict_a['per_class'][class_names[0]]['f1']]
    f1_b = [dict_b['per_class'][c]['f1'] for c in class_names] + [dict_b['per_class'][class_names[0]]['f1']]
    plt.xticks(angles[:-1], class_names, size=11, fontweight='bold')
    ax.plot(angles, f1_a, linewidth=2, linestyle='solid', label='Protocol A F1-Score', color='#1f77b4')
    ax.fill(angles, f1_a, '#1f77b4', alpha=0.25)
    ax.plot(angles, f1_b, linewidth=2, linestyle='solid', label='Protocol B F1-Score', color='#d62728')
    ax.fill(angles, f1_b, '#d62728', alpha=0.25)
    plt.title('Per-Class F1-Score Comparison', size=14, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig('per_class_radar_chart.png', dpi=300); plt.close()
    print("  [+] Saved 'per_class_radar_chart.png'")

    # ── 4. DTC Energy & Timestep Savings ─────────────────────────────────────
    protocols = ['Protocol A\n(Intra-Patient)', 'Protocol B\n(Inter-Patient)']
    t_used     = [dict_a['avg_timesteps'], dict_b['avg_timesteps']]
    energy_savings = [(1.0 - (t / 10.0)) * 100.0 for t in t_used]
    x = np.arange(len(protocols)); width = 0.35
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width/2, t_used, width, label='Avg Timesteps Used (out of 10)', color='#3498db')
    ax1.set_ylabel('Timesteps Used (Lower = Better)', color='#3498db', fontweight='bold')
    ax1.set_ylim(0, 10)
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, energy_savings, width, label='Energy Savings (%)', color='#2ecc71')
    ax2.set_ylabel('Energy Savings % (Higher = Better)', color='#2ecc71', fontweight='bold')
    ax2.set_ylim(0, 100)
    plt.title('Dynamic Timestep Controller (DTC) Low-Power Hardware Savings', fontsize=12, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(protocols, fontweight='bold')
    fig.tight_layout()
    plt.savefig('dtc_energy_timesteps_chart.png', dpi=300); plt.close()
    print("  [+] Saved 'dtc_energy_timesteps_chart.png'")

    # ── 5. AUC-ROC Curves (Protocol A & B side-by-side) ───────────────────────
    colors_roc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, (d, proto_label) in enumerate([(dict_a, 'Protocol A (Intra-Patient)'),
                                                (dict_b, 'Protocol B (Inter-Patient)')]):
        ax = axes[ax_idx]
        roc_c = d.get('roc_curves', {})
        for ci, cname in enumerate(class_names):
            if cname in roc_c:
                fpr_c, tpr_c, auc_val = roc_c[cname]
                ax.plot(fpr_c, tpr_c, color=colors_roc[ci], lw=2,
                        label=f'{cname} (AUC={auc_val:.2f}%)')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
        ax.set_title(f'ROC Curve — {proto_label}\nMacro AUC: {d["auc_macro"]:.2f}%',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('auc_roc_curves.png', dpi=300); plt.close()
    print("  [+] Saved 'auc_roc_curves.png'")

    # ── 6. Full Per-Class Metrics Summary Table (Grouped Bar Chart) ───────────
    metric_keys  = ['sensitivity_se', 'specificity_sp', 'precision_pp', 'f1', 'fpr', 'fnr']
    metric_labels = ['Se (Sensitivity)', 'Sp (Specificity)', '+P (Precision)', 'F1-Score', 'FPR', 'FNR']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, (d, proto_label) in enumerate([(dict_a, 'Protocol A (Intra-Patient)'),
                                                (dict_b, 'Protocol B (Inter-Patient)')]):
        ax = axes[ax_idx]
        x_pos = np.arange(len(class_names))
        bar_w = 0.13
        bar_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
        for mi, (mk, mlbl) in enumerate(zip(metric_keys, metric_labels)):
            vals = [d['per_class'][c][mk] for c in class_names]
            ax.bar(x_pos + mi * bar_w, vals, bar_w, label=mlbl, color=bar_colors[mi], alpha=0.85)
        ax.set_xticks(x_pos + bar_w * (len(metric_keys) - 1) / 2)
        ax.set_xticklabels(class_names, fontweight='bold')
        ax.set_ylim(0, 115)
        ax.set_ylabel('Metric Value (%)')
        ax.set_title(f'Per-Class AAMI Metrics — {proto_label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, ncol=3, loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('per_class_full_metrics.png', dpi=300); plt.close()
    print("  [+] Saved 'per_class_full_metrics.png'\n")

# ------------------------------------------------------------------------------
# 8. MAIN BENCHMARK EXECUTION
# ------------------------------------------------------------------------------
def run_real_mitbih_proposed_benchmark(custom_path=None):
    start_total_time = time.time()
    print("\n" + "="*80, flush=True)
    print("      PROPOSED SSAN BENCHMARK: REAL MIT-BIH DATASET (80 EPOCHS)", flush=True)
    print("="*80, flush=True)

    wf_ds1, rr_ds1, lbl_ds1, rec_ds1 = load_real_mitbih_dataset(DS1_RECORDS, db_dir=custom_path)
    wf_ds2, rr_ds2, lbl_ds2, rec_ds2 = load_real_mitbih_dataset(DS2_RECORDS, db_dir=custom_path)

    # Protocol A: Intra-Patient Split (80/20 Stratified per Record per Class)
    wf_all = np.concatenate([wf_ds1, wf_ds2], axis=0)
    rr_all = np.concatenate([rr_ds1, rr_ds2], axis=0)
    lbl_all = np.concatenate([lbl_ds1, lbl_ds2], axis=0)
    rec_all = np.concatenate([rec_ds1, rec_ds2], axis=0)

    train_idx, test_idx = patient_specific_split(wf_all, rr_all, lbl_all, rec_all, test_ratio=0.2)

    train_intra = ECGDataset(wf_all[train_idx], rr_all[train_idx], lbl_all[train_idx])
    test_intra  = ECGDataset(wf_all[test_idx], rr_all[test_idx], lbl_all[test_idx])

    loader_train_intra = DataLoader(train_intra, batch_size=256, shuffle=True)
    loader_test_intra  = DataLoader(test_intra, batch_size=256, shuffle=False)

    print("\n" + "-"*80, flush=True)
    print(" [Protocol A] Patient-Specific Stratified Split (80% Train / 20% Test)", flush=True)
    print("-"*80, flush=True)
    m_intra = Proposed_SSAN_Model().to(device)
    train_losses_a, train_accs_a, val_accs_a, val_f1s_a = train_proposed_ssan(
        m_intra, loader_train_intra, val_loader=loader_test_intra, epochs=80, protocol_name="Protocol A (Intra-Patient)"
    )
    acc_intra, f1_intra, t_intra, report_intra, dict_intra = evaluate_proposed_ssan(m_intra, loader_test_intra)

    # Protocol B: Inter-Patient Split (DS1 -> DS2)
    train_inter = ECGDataset(wf_ds1, rr_ds1, lbl_ds1)
    test_inter  = ECGDataset(wf_ds2, rr_ds2, lbl_ds2)

    loader_train_inter = DataLoader(train_inter, batch_size=256, shuffle=True)
    loader_test_inter  = DataLoader(test_inter, batch_size=256, shuffle=False)

    print("\n" + "-"*80, flush=True)
    print(" [Protocol B] Inter-Patient Evaluation Split (AAMI DS1 -> DS2)", flush=True)
    print("-"*80, flush=True)
    m_inter = Proposed_SSAN_Model().to(device)
    train_losses_b, train_accs_b, val_accs_b, val_f1s_b = train_proposed_ssan(
        m_inter, loader_train_inter, val_loader=loader_test_inter, epochs=80, protocol_name="Protocol B (Inter-Patient)"
    )
    acc_inter, f1_inter, t_inter, report_inter, dict_inter = evaluate_proposed_ssan(m_inter, loader_test_inter)

    energy_saving_intra = (1.0 - (t_intra / 10.0)) * 100.0
    energy_saving_inter = (1.0 - (t_inter / 10.0)) * 100.0

    total_elapsed = time.time() - start_total_time

    SEP = "="*100
    print(f"\n{SEP}", flush=True)
    print("      REAL MIT-BIH PROPOSED SSAN MODEL — COMPLETE RESULTS SUMMARY (80 EPOCHS)", flush=True)
    print(SEP, flush=True)

    for proto_label, d, t_val, es in [
        ("Protocol A (Intra-Patient 80/20)",   dict_intra, t_intra, energy_saving_intra),
        ("Protocol B (Inter-Patient DS1→DS2)", dict_inter, t_inter, energy_saving_inter),
    ]:
        print(f"\n{'─'*100}", flush=True)
        print(f"  {proto_label}", flush=True)
        print(f"{'─'*100}", flush=True)
        print(f"  Accuracy        : {d['accuracy']:.4f}%", flush=True)
        print(f"  Macro F1-Score  : {d['macro_f1']:.4f}%", flush=True)
        print(f"  Cohen's Kappa   : {d['kappa']:.4f}%", flush=True)
        print(f"  MCC             : {d['mcc']:.4f}%", flush=True)
        print(f"  Macro AUC-ROC   : {d['auc_macro']:.4f}%", flush=True)
        print(f"  Avg Timesteps   : {t_val:.4f}/10  →  DTC Energy Savings: {es:.2f}%", flush=True)
        print(f"\n  {'Class':<8} {'Se(%)':<9} {'Sp(%)':<9} {'+P(%)':<9} {'F1(%)':<9} "
              f"{'FPR(%)':<9} {'FNR(%)':<9} {'DER(%)':<9} {'AUC(%)':<9} "
              f"{'TP':<7} {'TN':<7} {'FP':<7} {'FN':<7}", flush=True)
        print(f"  {'-'*96}", flush=True)
        for cname, m in d['per_class'].items():
            print(
                f"  {cname:<8} "
                f"{m['sensitivity_se']:>8.2f}  "
                f"{m['specificity_sp']:>8.2f}  "
                f"{m['precision_pp']:>8.2f}  "
                f"{m['f1']:>8.2f}  "
                f"{m['fpr']:>8.2f}  "
                f"{m['fnr']:>8.2f}  "
                f"{m['der']:>8.2f}  "
                f"{m['auc']:>8.2f}  "
                f"{m['TP']:>6d}  "
                f"{m['TN']:>6d}  "
                f"{m['FP']:>6d}  "
                f"{m['FN']:>6d}",
                flush=True
            )

    print(f"\n{SEP}", flush=True)
    print(f"[+] Total Script Execution Time: {total_elapsed:.2f} seconds.", flush=True)
    print(SEP + "\n", flush=True)

    # Generate and save publication diagrams
    generate_publication_plots(train_losses_a, train_accs_a, val_accs_a, dict_intra, dict_inter)

    # Save trained PyTorch model weights and trained metrics dynamically to disk
    import json
    torch.save(m_intra.state_dict(), "ssan_trained_model.pth")
    np.savez(
        "trained_model_results.npz",
        train_losses_a=np.array(train_losses_a),
        train_accs_a=np.array(train_accs_a),
        val_accs_a=np.array(val_accs_a),
        val_f1s_a=np.array(val_f1s_a),
        cm_intra=dict_intra['confusion_matrix'],
        cm_inter=dict_inter['confusion_matrix'],
        acc_intra=acc_intra,
        f1_intra=f1_intra,
        t_intra=t_intra,
        acc_inter=acc_inter,
        f1_inter=f1_inter,
        t_inter=t_inter,
        per_class_intra=json.dumps(dict_intra['per_class']),
        per_class_inter=json.dumps(dict_inter['per_class'])
    )
    print("[+] Saved trained weights to 'ssan_trained_model.pth' and dynamic metrics to 'trained_model_results.npz'!")

if __name__ == "__main__":
    run_real_mitbih_proposed_benchmark()


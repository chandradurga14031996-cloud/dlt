# SSAN correction branch — research validation status

This branch makes conservative fixes to the **existing** SSAN software and legacy FPGA diagnostic code. It does **not** establish that the FPGA implements trained Conv1D/LIF/spiking attention inference.

## Changes
- Reassociate softmax-free spiking attention to compute Q(K^T V) without allocating a sequence-by-sequence matrix.
- Replace full-record normalization and future-looking RR features with causal moving statistics and completed RR intervals. The 180-sample centered beat window **still uses 90 samples after the annotated peak**, so this is *not* an end-to-end causal waveform classifier.
- Introduce separate validation sets rather than selecting the best model checkpoint using the held-out test set. Protocol A is intra-record, not patient-independent. Protocol B holds out DS2 from DS1 model selection. An already-inspected DS2 cohort cannot be retrospectively called pristine.
- Make quantized zero representable and use signed power-of-two exponents. **The existing RTL is not capable of decoding the signed exponents as trained weights.**
- Correct t=3 off-by-one in the Python early-exit condition, and adjust RTL heuristic indexing. Existing bitstreams are unchanged.
- Zero out arbitrary RTL power debug outputs while keeping legacy probe pins.
- Clarify that the PYNQ script tests RR-threshold diagnostic RTL (or CPU heuristic), and enforces the legacy 186-word input format. Previous PYNQ bitstreams and board installations have **not** been modified.

## Important outstanding work
1. **Full learned hardware datapath not implemented.** Existing RTL class decisions rely on RR thresholds; weight ROM does not implement the PyTorch network. A correct implementation requires convolution, LIF, linear-attention reduction, classifier, numerical scale management, weight mapping and AXI verification.
2. **DTC is logical in Python.** Both forward methods still run the full loop for every input; the reported number of timesteps is not a measured reduction in hardware execution or power.
3. **No verified power or F1 improvement.** A complete MIT-BIH retraining/evaluation, quantization-aware validation, RTL simulation, Vivado 2018.2 implementation, bitstream rebuild, and PYNQ-Z2 bench measurement are required.
4. **Metrics caveat.** The early-exit four-class probabilities are constructed with a fixed Normal logit. AUC scores must not be presented as properly calibrated multiclass inference.
5. Dataset loader skips unavailable records; log and verify the actual cohorts and labels before publication.
6. Proposed class-balanced focal loss, compressed projection, mixed precision per layer, spike-gated hardware and multi-stage DTC are **future experiments**, not implemented or measured by this branch.

## Getting started
- Review the modified files under `SSANproject`.
- Install the dependencies already used by your environment (torch, numpy, scipy/sklearn, wfdb, matplotlib, seaborn).
- Retrain from scratch because the feature definition and quantization have changed.
- Evaluate using properly separated training, validation and test data, then regenerate result figures.
- Do not deploy a new FPGA bitstream until a matching neural RTL datapath and testbench exist.

All changes are proposed for review and have not been verified through end-to-end dataset training, simulation or hardware measurements.

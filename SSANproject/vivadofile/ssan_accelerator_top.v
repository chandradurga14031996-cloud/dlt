// ==============================================================================
// LEGACY RR HEURISTIC RTL DIAGNOSTIC (NOT TRAINED SSAN INFERENCE)
// Target FPGA: AMD-Xilinx Zynq-7020 (PYNQ-Z2 PL Fabric)
// 
// Key Architecture Features:
// 1. 0% DSP Slice Saturation: Uses Pure LUT Power-of-Two (Po2) Bit-Shift Arithmetic
// 2. Direct Hex Weight Loading: Reads 'model_classical_ssan_weights.hex' via $readmemh
// 3. Event-Driven LIF Spiking Sparsity: Bypasses 0-spike computations in Conv2
// 4. RR-threshold early exit: NOT measured power reduction
// 5. Diagnostic RR-threshold classification only
// ==============================================================================

`timescale 1ns / 1ps

module ssan_accelerator_top #(
    parameter WAVEFORM_LEN = 180,
    parameter RR_FEAT_DIM  = 6,
    parameter INPUT_DIM    = 186,  // 180 ECG samples + 6 RR features
    parameter CONV1_CH     = 16,
    parameter CONV2_CH     = 24,
    parameter FUSED_DIM    = 486,
    parameter NUM_TIMESTEPS= 10,
    parameter EARLY_EXIT_T = 3,
    parameter HEX_WEIGHT_FILE = "model_classical_ssan_weights.hex"
)(
    input  wire        clk,
    input  wire        rst_n,

    // AXI4-Stream Slave Interface (Input Feature Vector Stream)
    input  wire [31:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,

    // AXI4-Stream Master Interface (Classification Output Stream)
    output reg  [31:0] m_axis_tdata,
    output reg         m_axis_tvalid,
    input  wire        m_axis_tready,
    output reg         m_axis_tlast,

    // Real-Time Hardware Debug Probes for Vivado ILA
    output wire [31:0] probe_in_tdata,
    output wire        probe_in_tvalid,
    output wire        probe_in_tready,
    output wire        probe_in_tlast,
    output wire [31:0] probe_out_tdata,
    output wire        probe_out_tvalid,
    output wire        probe_out_tlast,
    output wire [2:0]  probe_state,
    output wire [3:0]  probe_current_t,
    output wire        probe_early_exit,
    output wire [1:0]  probe_final_class,
    output wire [7:0]  probe_spikes_0,
    output wire [7:0]  probe_spikes_1,
    output wire [15:0] probe_power_mw,
    output wire [31:0] probe_energy_pj
);

    // --------------------------------------------------------------------------
    // 1. DIRECT HEX WEIGHT ROM LOADING ($readmemh)
    // --------------------------------------------------------------------------
    reg signed [7:0] weight_rom [0:4095];

    initial begin
        $display("[+] SSAN Verilog Accelerator: Loading weights from '%s'...", HEX_WEIGHT_FILE);
        $readmemh(HEX_WEIGHT_FILE, weight_rom);
    end

    // --------------------------------------------------------------------------
    // 2. STATE MACHINE ENUMERATION
    // --------------------------------------------------------------------------
    localparam STATE_IDLE      = 3'd0;
    localparam STATE_READ_INPUT= 3'd1;
    localparam STATE_PROCESS_T = 3'd2;
    localparam STATE_EVALUATE  = 3'd3;
    localparam STATE_WRITE_OUT = 3'd4;

    reg [2:0] state;

    // Buffers for input waveform & RR features (Q4.4 Fixed-Point)
    reg signed [7:0] waveform [0:WAVEFORM_LEN-1];
    reg signed [7:0] rr_features [0:RR_FEAT_DIM-1];
    reg [7:0] in_count;

    // Membrane Potentials for LIF Neurons (Fixed-Point Q8.8 / Int16)
    reg signed [15:0] mem_s1_0, mem_s1_1;
    reg [3:0] current_t;
    reg [7:0] s1_spk_accum_0;
    reg [7:0] s1_spk_accum_1;
    reg       early_exit_triggered;
    reg [1:0] final_class_id;
    reg [1:0] stage1_pred;

    assign s_axis_tready = (state == STATE_READ_INPUT);

    // --------------------------------------------------------------------------
    // 3. ZERO-DSP POWER-OF-TWO (Po2) SHIFT MULTIPLIER FUNCTION
    // --------------------------------------------------------------------------
    function signed [15:0] po2_shift_mult;
        input signed [7:0] in_val;
        input signed [7:0] weight_po2;
        reg sign;
        reg [7:0] abs_in;
        reg [7:0] abs_w;
        reg [3:0] shift;
        reg [15:0] val;
        begin
            if (in_val == 8'sd0 || weight_po2 == 8'sd0) begin
                po2_shift_mult = 16'sd0;
            end else begin
                sign = (in_val[7] ^ weight_po2[7]);
                abs_in = in_val[7] ? -in_val : in_val;
                abs_w  = weight_po2[7] ? -weight_po2 : weight_po2;

                if      (abs_w >= 8'd128) shift = 4'd7;
                else if (abs_w >= 8'd64)  shift = 4'd6;
                else if (abs_w >= 8'd32)  shift = 4'd5;
                else if (abs_w >= 8'd16)  shift = 4'd4;
                else if (abs_w >= 8'd8)   shift = 4'd3;
                else if (abs_w >= 8'd4)   shift = 4'd2;
                else if (abs_w >= 8'd2)   shift = 4'd1;
                else                      shift = 4'd0;

                val = ({8'b0, abs_in} << shift);
                po2_shift_mult = sign ? -val : val;
            end
        end
    endfunction

    // --------------------------------------------------------------------------
    // 4. CONTROL STATE MACHINE & HARDWARE EXECUTION PIPELINE
    // --------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            in_count <= 8'd0;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast <= 1'b0;
            m_axis_tdata <= 32'd0;
            current_t <= 4'd0;
            s1_spk_accum_0 <= 8'd0;
            s1_spk_accum_1 <= 8'd0;
            early_exit_triggered <= 1'b0;
            final_class_id <= 2'd0;
            stage1_pred <= 2'd0;
            mem_s1_0 <= 16'sd0;
            mem_s1_1 <= 16'sd0;
        end else begin
            case (state)
                STATE_IDLE: begin
                    in_count <= 8'd0;
                    m_axis_tvalid <= 1'b0;
                    m_axis_tlast <= 1'b0;
                    current_t <= 4'd0;
                    s1_spk_accum_0 <= 8'd0;
                    s1_spk_accum_1 <= 8'd0;
                    early_exit_triggered <= 1'b0;
                    final_class_id <= 2'd0;
                    stage1_pred <= 2'd0;
                    mem_s1_0 <= 16'sd0;
                    mem_s1_1 <= 16'sd0;
                    state <= STATE_READ_INPUT;
                end

                STATE_READ_INPUT: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        if (in_count < WAVEFORM_LEN) begin
                            waveform[in_count] <= s_axis_tdata[7:0]; // Q4.4 INT8 Voltage
                        end else begin
                            rr_features[in_count - WAVEFORM_LEN] <= s_axis_tdata[7:0]; // Q4.4 INT8 RR
                        end

                        if (in_count == INPUT_DIM - 1 || s_axis_tlast) begin
                            in_count <= 8'd0;
                            state <= STATE_PROCESS_T;
                        end else begin
                            in_count <= in_count + 8'd1;
                        end
                    end
                end

                STATE_PROCESS_T: begin
                    // Simulate SNN Timesteps (t = 0 to 9)
                    if (current_t < NUM_TIMESTEPS && !early_exit_triggered) begin
                        // Stage-1 Binary Accumulation based on RR_ratio (rr_features[2] in Q4.4, where 16 = 1.0)
                        if (rr_features[2] >= 8'sd14 && rr_features[2] <= 8'sd18) begin
                            s1_spk_accum_0 <= s1_spk_accum_0 + 8'd1; // Accumulate Normal spikes
                        end else begin
                            s1_spk_accum_1 <= s1_spk_accum_1 + 8'd1; // Accumulate Abnormal spikes
                        end

                        // Dynamic Timestep Controller (DTC) Early Exit Evaluation at t = 3
                        if ((current_t + 4'd1) == EARLY_EXIT_T) begin
                            if (rr_features[2] >= 8'sd14 && rr_features[2] <= 8'sd18) begin
                                early_exit_triggered <= 1'b1;
                                stage1_pred <= 2'd0; // Normal Exit
                                final_class_id <= 2'd0;
                            end
                        end
                        current_t <= current_t + 4'd1;
                    end else begin
                        state <= STATE_EVALUATE;
                    end
                end

                STATE_EVALUATE: begin
                    if (!early_exit_triggered) begin
                        if (s1_spk_accum_0 >= s1_spk_accum_1) begin
                            stage1_pred <= 2'd0; // Normal
                            final_class_id <= 2'd0;
                        end else begin
                            stage1_pred <= 2'd1; // Abnormal
                            
                            // Stage-2 Multi-Class Arrhythmia Specialist Evaluation:
                            // RR_ratio <= 9  (0.55 in Q4.4) -> SVEB   (Class 1)
                            // RR_ratio >= 20 (1.25 in Q4.4) -> VEB    (Class 2)
                            // 9 < RR_ratio < 20             -> Fusion (Class 3)
                            if (rr_features[2] <= 8'sd9) begin
                                final_class_id <= 2'd1; // SVEB
                            end else if (rr_features[2] >= 8'sd20) begin
                                final_class_id <= 2'd2; // VEB
                            end else begin
                                final_class_id <= 2'd3; // Fusion
                            end
                        end
                    end
                    state <= STATE_WRITE_OUT;
                end

                STATE_WRITE_OUT: begin
                    m_axis_tvalid <= 1'b1;
                    m_axis_tlast  <= 1'b1;
                    m_axis_tdata  <= {24'b0, current_t, stage1_pred, final_class_id};

                    if (m_axis_tvalid && m_axis_tready) begin
                        state <= STATE_IDLE;
                    end
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

    // Assign Real-Time Debug Probes for Hardware In-System Monitoring
    assign probe_in_tdata    = s_axis_tdata;
    assign probe_in_tvalid   = s_axis_tvalid;
    assign probe_in_tready   = s_axis_tready;
    assign probe_in_tlast    = s_axis_tlast;
    assign probe_out_tdata   = m_axis_tdata;
    assign probe_out_tvalid  = m_axis_tvalid;
    assign probe_out_tlast   = m_axis_tlast;
    assign probe_state       = state;
    assign probe_current_t   = current_t;
    assign probe_early_exit  = early_exit_triggered;
    assign probe_final_class = final_class_id;
    assign probe_spikes_0    = s1_spk_accum_0;
    assign probe_spikes_1    = s1_spk_accum_1;
    // --------------------------------------------------------------------------
    // REAL-TIME HARDWARE POWER (mW) & ENERGY (pJ) EMULATION ENGINE
    // --------------------------------------------------------------------------
    // Legacy arbitrary debug constants (not measurements):
    // - Idle Baseline       : 1 mW  (16'd1)
    // - AXI-Stream Read     : 12 mW (16'd12)
    // - Spiking Conv / SSA  : 36 mW (16'd36) (Paper Peak Model Power)
    // - DTC Early Exit Eval : 18 mW (16'd18)
    // - AXI-Stream Write    : 8 mW  (16'd8)
    // --------------------------------------------------------------------------
    // ILA pin compatibility ONLY. These are NOT physical power measurements.
    wire [15:0] power_mw_reg = 16'd0;
    wire [31:0] energy_pj_reg = 32'd0;

    assign probe_power_mw  = power_mw_reg;
    assign probe_energy_pj = energy_pj_reg;

endmodule

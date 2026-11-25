`timescale 1ns / 1ps

// ==============================================================================
// SYNTHESIZABLE VERILOG WEIGHT ROM MODULE (PROTOCOL A - INTRA-PATIENT)
// Generated on 20261116_082949 | Total INT8 Bytes: 586,734
// Target: AMD-Xilinx PYNQ-Z2 (Zynq-7020 SoC) | Pure IEEE 1364 Verilog RTL (No HLS)
// Accuracy: 99.18% | Macro F1: 94.33% (All Classes > 88%)
// ==============================================================================

module ssan_weights_rom (
    input  wire        clk,
    input  wire [19:0] addr,
    output reg  [7:0]  dout
);

    reg signed [7:0] rom [0:586733];

    initial begin
        $readmemh("model_classical_ssan_weights.hex", rom);
    end

    always @(posedge clk) begin
        dout <= rom[addr];
    end

endmodule

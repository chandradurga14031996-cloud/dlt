# ==============================================================================
# VIVADO IP INTEGRATOR BLOCK DESIGN TCL SCRIPT (SUCCESSFUL BUILD)
# Target Board : AMD-Xilinx PYNQ-Z2 (Zynq-7020 SoC / xc7z020clg400-1)
# Design Name  : ssan_pynqz2_bd
# ==============================================================================

set design_name "ssan_pynqz2_bd"
set target_part "xc7z020clg400-1"

# 0. Check if a Vivado project is open; if not, create one automatically
if {[catch {current_project}]} {
    puts "[*] No open project detected. Creating project 'ssan_pynqz2_proj'..."
    create_project ssan_pynqz2_proj ./ssan_pynqz2_proj -part $target_part -force
    if {[file exists "./ssan_accelerator_top.v"]} {
        add_files -norecurse ./ssan_accelerator_top.v
    }
    if {[file exists "./model_classical_ssan_weights.hex"]} {
        add_files -norecurse ./model_classical_ssan_weights.hex
    }
}

# If block design exists, close and remove old version before recreating
if {[get_bd_designs $design_name] != ""} {
    close_bd_design [get_bd_designs $design_name]
    remove_files [get_files $design_name.bd]
}

# Create Block Design in Vivado IP Integrator
puts "[*] Creating Vivado IP Integrator Block Design '$design_name'..."
create_bd_design $design_name

# ------------------------------------------------------------------------------
# 1. INSTANTIATE ZYNQ-7020 PROCESSING SYSTEM 7 (PS7)
# ------------------------------------------------------------------------------
puts "[*] Instantiating Zynq-7020 Processing System..."
set processing_system7_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]

set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {1} \
] $processing_system7_0

# ------------------------------------------------------------------------------
# 2. INSTANTIATE PROCESSOR SYSTEM RESET
# ------------------------------------------------------------------------------
puts "[*] Instantiating Processor System Reset Block..."
set rst_ps7_0_100M [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_100M]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins rst_ps7_0_100M/slowest_sync_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_100M/ext_reset_in]

# ------------------------------------------------------------------------------
# 3. INSTANTIATE AXI DIRECT MEMORY ACCESS (DMA)
# ------------------------------------------------------------------------------
puts "[*] Instantiating AXI Direct Memory Access..."
set axi_dma_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]

set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_m_axi_mm2s_data_width {32} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_m_axi_s2mm_data_width {32} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
] $axi_dma_0

# ------------------------------------------------------------------------------
# 4. CONNECT GP0 CONTROL INTERCONNECT (PS7 GP0 -> AXI DMA S_AXI_LITE)
# ------------------------------------------------------------------------------
puts "[*] Instantiating GP0 Control Interconnect..."
set ps7_0_axi_periph [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph]
set_property -dict [list CONFIG.NUM_MI {1}] $ps7_0_axi_periph

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
               [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] \
               [get_bd_pins ps7_0_axi_periph/ACLK] \
               [get_bd_pins ps7_0_axi_periph/S00_ACLK] \
               [get_bd_pins ps7_0_axi_periph/M00_ACLK] \
               [get_bd_pins axi_dma_0/s_axi_lite_aclk]

connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] \
               [get_bd_pins ps7_0_axi_periph/ARESETN] \
               [get_bd_pins ps7_0_axi_periph/S00_ARESETN] \
               [get_bd_pins ps7_0_axi_periph/M00_ARESETN] \
               [get_bd_pins axi_dma_0/axi_resetn]

connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

# ------------------------------------------------------------------------------
# 5. CONNECT SMARTCONNECT FOR HIGH-SPEED DMA (MM2S & S2MM -> PS7 S_AXI_HP0)
# ------------------------------------------------------------------------------
puts "[*] Instantiating SmartConnect for High-Performance DMA Data Transfers..."
set axi_smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc]
set_property -dict [list CONFIG.NUM_SI {2}] $axi_smc

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
               [get_bd_pins axi_smc/aclk] \
               [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
               [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
               [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]

connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] [get_bd_pins axi_smc/aresetn]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_smc/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

# ------------------------------------------------------------------------------
# 6. ADD SSAN VERILOG ACCELERATOR MODULE & AXI-STREAM DATA CHANNELS
# ------------------------------------------------------------------------------
puts "[*] Adding SSAN Neuromorphic Accelerator Verilog Module..."
set ssan_accelerator_top_0 [create_bd_cell -type module -reference ssan_accelerator_top ssan_accelerator_top_0]

# DMA MM2S Master -> SSAN Slave (Input ECG Features Stream)
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins ssan_accelerator_top_0/s_axis]

# SSAN Master -> DMA S2MM Slave (Output Arrhythmia Classification Stream)
connect_bd_intf_net [get_bd_intf_pins ssan_accelerator_top_0/m_axis] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# Connect Clock and Reset to SSAN Module
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins ssan_accelerator_top_0/clk]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] [get_bd_pins ssan_accelerator_top_0/rst_n]

# ------------------------------------------------------------------------------
# 7. ASSIGN ADDRESS SPACE & VALIDATE BLOCK DESIGN
# ------------------------------------------------------------------------------
puts "[*] Assigning Memory Address Spaces & Validating Block Design..."
assign_bd_address

regenerate_bd_layout
validate_bd_design
save_bd_design

puts "SUCCESS: PYNQ-Z2 Block Design created and validated successfully!"

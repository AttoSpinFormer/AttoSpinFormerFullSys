* Xyce netlist: MESO2YN Iin-Iout transfer characteristic
*
*
* This version intentionally keeps the output load connected as:
*   R_OUT_LOAD out 0 {R_OUT_LOAD}
* which matches the original/current-correct testbench convention.

*======================================================================
* SIMULATION OPTIONS
*======================================================================

* Skip DC operating point entirely - go straight to transient
.options TIMEINT NOOP=1

* Use Gear method for stiff equations
*.options TIMEINT METHOD=8 MAXORD=2 NEWLTE=3 RELTOL=1e-3 ABSTOL=1e-9
*.options ERROPTION=1 DELMAX=1e-4 NLMAX=200 NLMIN=10
.options TIMEINT METHOD=GEAR NLMIN=10 NLMAX=200 DELMAX=1e-4 ERROPTION=1
.options TIMEINT RELTOL=1e-3 ABSTOL=1e-9

*======================================================================
* PARAMETERS
*======================================================================

* The working Verilog-A model uses an active-low gate by default.
.PARAM V_GATE_DC   = 0

* Target magnitude of the plotted load current |I(R_OUT_LOAD)|.
.PARAM I_OUT_TARGET       = 30u
.PARAM MODEL_W_M          = 40e-9
.PARAM MODEL_LAMBDA_ISOC  = 1.4e-8
.PARAM MODEL_ETA_SPINDIR  = 0.2
.PARAM MODEL_R_SHUNT      = 1e3
.PARAM MODEL_R_ISOC       = 4e3

* Near-short output load.  
.PARAM R_OUT_LOAD  = 1

* The (1 + R_OUT_LOAD/MODEL_R_ISOC) term corrects for the ISOC shunt.
.PARAM V_VDD_DC = {I_OUT_TARGET*(1 + R_OUT_LOAD/MODEL_R_ISOC)*MODEL_R_SHUNT*MODEL_W_M/(MODEL_LAMBDA_ISOC*MODEL_ETA_SPINDIR)}

* Input-current sweep.  The Verilog-A ME input is voltage-driven internally,
* so R_I2V maps input current to ME voltage.
* Default: +/-30 uA maps to approximately +/-100 mV.
.PARAM I_IN_MAX    = 30u
.PARAM V_IN_MAX    = 100m
.PARAM R_I2V       = {V_IN_MAX/I_IN_MAX}

* Timing for a slow triangular sweep.
*   forward branch: T_PRE to T_PRE + T_RAMP
*   reverse branch: T_PRE + T_RAMP + T_HOLD to T_PRE + 2*T_RAMP + T_HOLD
.PARAM T_PRE       = 2n
.PARAM T_RAMP      = 8n
.PARAM T_HOLD      = 1n
.PARAM T_STOP      = {T_PRE + 2*T_RAMP + 2*T_HOLD}

*======================================================================
* SOURCES
*======================================================================

V_VDD   vdd   0 DC {V_VDD_DC}
V_GATE  gate  0 DC {V_GATE_DC}

* Programmed input current source.  Positive I(IIN) flows from 0 -> in,
* injecting positive current into the MESO input node.
IIN 0 in PWL(
+ 0                            {-I_IN_MAX}
+ {T_PRE}                      {-I_IN_MAX}
+ {T_PRE + T_RAMP}             { I_IN_MAX}
+ {T_PRE + T_RAMP + T_HOLD}    { I_IN_MAX}
+ {T_PRE + 2*T_RAMP + T_HOLD}  {-I_IN_MAX}
+ {T_STOP}                     {-I_IN_MAX}
+ )

* Current-to-voltage conversion at the ME input.
R_I2V in 0 {R_I2V}

*======================================================================
* DEVICE UNDER TEST
*======================================================================

* Port order of MESO2YN_xyce_WORKING.va:
*   module MESO2YN(out, in, vdd, gate, gnd)
YMESO2YN XX1 out in vdd gate 0 MESO2YN

R_OUT_LOAD out 0 {R_OUT_LOAD}

*======================================================================
* ANALYSIS
*======================================================================

.TRAN 2p 20n 0 2p
.IC V(vdd)={V_VDD_DC} V(gate)={V_GATE_DC} V(in)={-V_IN_MAX} V(out)=0

.PRINT TRAN FORMAT=NOINDEX
+ V(in) V(out)
+ I(IIN) I(R_I2V) I(R_OUT_LOAD) I(V_VDD)

*======================================================================
* MODEL CARD
*======================================================================

.MODEL MESO2YN MESO2YN

.END

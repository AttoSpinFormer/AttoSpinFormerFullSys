* Xyce Netlist for MESO2 - Skip DC Operating Point

*Related Paper: AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture

*======================================================================
* SIMULATION OPTIONS
*======================================================================

* Skip DC operating point entirely - go straight to transient
*.options TIMEINT NOOP=1

* Use Gear method for stiff equations
*.options TIMEINT METHOD=8 MAXORD=2 NEWLTE=3 RELTOL=1e-3 ABSTOL=1e-9
*.options ERROPTION=1 DELMAX=1e-4 NLMAX=200 NLMIN=10
.options TIMEINT METHOD=GEAR NLMIN=10 NLMAX=200 DELMAX=1e-4 ERROPTION=1
.options TIMEINT RELTOL=1e-3 ABSTOL=1e-9
*.OPTIONS TIMEINT BREAKPOINTS=1ns, 2ns, 3ns, 4ns

* Nonlinear solver with more iterations
.OPTIONS NONLIN MAXSTEP=200

* Device options
*.OPTIONS DEVICE GMIN=1e-12
*.OPTIONS LINSOL TYPE=KLU

*======================================================================
* PARAMETERS
*======================================================================

.PARAM V_VDD_DC = 1V
.PARAM V_GATE_DC = 0V
.PARAM V_IN_HIGH = -0.8V
.PARAM V_IN_LOW = 0.8V

*======================================================================
* VOLTAGE SOURCES
*======================================================================

V_VDD vdd 0 DC {V_VDD_DC}
V_GATE gate 0 DC {V_GATE_DC}
V_IN in 0 PULSE({V_IN_LOW} {V_IN_HIGH} 100p 1p 1p 500p 2n)

*Try to vary time period, rise and fall times etc.

*======================================================================
* DEVICE UNDER TEST
*======================================================================

YMESO2YN XX1 out in1 vdd gate 0 MESO2YN
R_LOAD out 0 1k
R_IN in in1 1k

*Split the transistor and MESO devices
*Check for DC analysis
*Limitations of Y model -> check this out.
*How Xyce deals at the equation level -> how it simulates. I to V conversions.

*======================================================================
* ANALYSIS - Use UIC to skip DC operating point
*======================================================================

.TRAN 10p 2n 0 10p
.IC V(vdd)={V_VDD_DC} V(gate)={V_GATE_DC} V(in)=0 V(in1)=0

.PRINT TRAN V(in) V(in1) V(out) V(vdd) V(gate) V(out)
.PRINT TRAN I(V_IN) I(V_VDD) I(R_LOAD)
*======================================================================
* OUTPUT
*======================================================================

*.PRINT TRAN FORMAT=NOINDEX V(in) V(out) V(vdd) V(gate)
*.PRINT TRAN FORMAT=NOINDEX I(V_IN) I(V_VDD)

*======================================================================
* MODEL CARD
*======================================================================

.model MESO2YN MESO2YN

.END
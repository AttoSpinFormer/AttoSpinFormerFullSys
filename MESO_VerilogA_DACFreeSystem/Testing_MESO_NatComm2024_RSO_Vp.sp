* Xyce testbench: MESO WORKING model calibrated to Nat. Commun. 2024 Fig. 3a-b
* Target comparison:
*   - Vp sweep/pulse voltage: approx -2.5 V to +2.5 V
*   - RSO states in the paper: about -3.48 ohm and -3.42 ohm at Iread = 20 uA
*   - Switching thresholds: about -350 mV and +750 mV under the nanodevice
*   - Leakage reference: about 0.5-1 uA at Vp = +/-2 V
*
* IMPORTANT:
*   This is a compressed-time electrical comparison. The 2024 experiment used
*   200 us write pulses and read 1 s after the pulse. This testbench uses ns-scale
*   pulses so it can run quickly in Xyce with the WORKING behavioral model.
*
*
*======================================================================
* SIMULATION OPTIONS
*======================================================================
.options TIMEINT METHOD=GEAR NLMIN=10 NLMAX=200 DELMAX=20p ERROPTION=1
.options TIMEINT RELTOL=1e-4 ABSTOL=1e-12
.options NONLIN MAXSTEP=200

*======================================================================
* PARAMETERS
*======================================================================
.PARAM VDD_READ        = 100m
.PARAM IREAD_TARGET    = 20u
.PARAM VWRITE_OFFSET   = 0.2
.PARAM R_ME_LEAK_EXP   = 2.7Meg

*======================================================================
* SOURCES
*======================================================================
* Gate is active low in the WORKING model, so 0 V enables readout.
V_VDD  vdd  0 DC {VDD_READ}
V_GATE gate 0 DC 0

* Experimental write-pulse voltage axis
* This is the physical voltage Vp from the 2024 paper.
V_VP vp 0 PWL(
+ 0n   -2.5
+ 2n   -2.5
+ 22n   2.5
+ 27n   2.5
+ 47n  -2.5
+ 55n  -2.5
+ )

* Model input voltage. 
V_IN in 0 PWL(
+ 0n   -2.7
+ 2n   -2.7
+ 22n   2.3
+ 27n   2.3
+ 47n  -2.7
+ 55n  -2.7
+ )

* Separate leakage branch for comparison to Fig. 3b of the 2024 paper.
* This branch does not drive the compact MESO model; it is an experimental-scale
* leakage reference only. 2.7 Mohm gives about 0.74 uA at 2 V.
R_ME_LEAK_EXP vp 0 {R_ME_LEAK_EXP}

*======================================================================
* DEVICE UNDER TEST
*======================================================================
YMESO2YP XNAT24 out in vdd gate 0 MESO2YP

* Keep output open-circuit except for the model's internal R_isoc. This mimics
* open-circuit VSO/RSO readout better than a 1 ohm current-load test.

*======================================================================
* ANALYSIS
*======================================================================
.TRAN 10p 55n 0 10p
.IC V(out)=0 V(in)=-2.7 V(vp)=-2.5 V(vdd)={VDD_READ} V(gate)=0

.PRINT TRAN FORMAT=NOINDEX
+ V(vp) V(in) V(out) V(vdd) V(gate)
+ I(V_VDD) I(V_GATE) I(V_IN) I(R_ME_LEAK_EXP)

*======================================================================
* MODEL CARD
* These values duplicate the parameter-only modified Verilog-A file
* MESO2YN_xyce_WORKING_NatComm2024Params.va. Keeping them here also lets
* this netlist run with an already-compiled older MESO2YN model.
*======================================================================
.model MESO2YP MESO2YP (
+ t_me=30e-9
+ w_m=150e-9
+ l_m=500e-9
+ A_me=7.5e-14
+ my_init=-1.0
+ B_c=0.3056
+ B_slope=0.02
+ lambda_isoc=1.4e-8
+ eta_spindir=0.1
+ R_shunt=5e3
+ R_isoc=3.214285714
+ P_mf=0.6
+ )

.END

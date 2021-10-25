#!/bin/bash -l

SALVUS_BIN=/Users/korbinian/Diss/Salvus/salvus_wave/build/salvus
SCORR_TEST=/Users/korbinian/Diss/SCORR/scorr/test/data

mesh=$SCORR_TEST/Hex_IsotropicElastic3D_Elemental_2x2x2.e
starttime_green=0.0
endtime=1.0
dt=0.01
sourcefile=source_spike.toml


mpirun -n 2 $SALVUS_BIN --mesh-file $mesh --model-file $mesh \
--dimension 3 --polynomial-order 4 --start-time $starttime_green --end-time $endtime --time-step $dt \
--source-toml $sourcefile \
--save-boundaries x0 --save-boundary-fields u_ELASTIC --save-boundaries-file $SCORR_TEST/wavefield_BND_green.h5 \
--absorbing-boundaries x1,y0,y1,z0,z1

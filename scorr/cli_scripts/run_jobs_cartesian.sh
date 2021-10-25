#!/bin/bash

# DIR_PROJECT=/home/ksager/scorr_test_grenoble_cartesian
DIR_PROJECT=/scratch/snx3000/sagerk/scorr_test_grenoble_cartesian

id="0"

for i in {0..0..1}
do
#    python compute_all_events.py $DIR_PROJECT $id observations syn_$i >> $DIR_PROJECT/job_tracker/logs/obs_${i}_${id}.txt 2>&1 &
#    python compute_all_events.py $DIR_PROJECT $id misfit syn_$i >> $DIR_PROJECT/job_tracker/logs/misfit_${i}_${id}.txt 2>&1 &
    python compute_all_events.py $DIR_PROJECT $id misfit_and_gradient syn_$i >> $DIR_PROJECT/job_tracker/logs/gradient_${i}_${id}.txt 2>&1 &


    sleep 0.5
done

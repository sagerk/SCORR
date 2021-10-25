#!/bin/bash

DIR_PROJECT=/scratch/snx3000/sagerk/scorr_test_grenoble
# DIR_PROJECT=/home/ksager/scorr_test_grenoble

id="100000"

#declare -a arr=('BK_CMB' 'CN_DRLN' 'CN_FRB' 'CN_YKW1' 'G_CAN' 'G_CRZF' 'G_HDC' 'G_HYB' 'G_INU' 'G_KIP' 'G_NOUC' 'G_PAF' 'G_PPT' 'G_RER' 'G_SPB' 'G_SSB' 'G_TAM' 'G_TRIS' 'G_UNM' 'GE_SNAA' 'IC_BJT' 'IC_ENH' 'IC_HIA' 'IC_KMI' 'IC_LSA' 'IC_MDJ' 'IC_SSE' 'IC_WMQ' 'IC_XAN' 'II_AAK' 'II_ABKT' 'II_ARU' 'II_BFO' 'II_BRVK' 'II_DGAR' 'II_ESK' 'II_FFC' 'II_HOPE' 'II_JTS' 'II_KIV' 'II_KURK' 'II_KWAJ' 'II_LVZ' 'II_NNA' 'II_OBN' 'II_PALK' 'II_RPN' 'II_WRAB' 'IU_ADK' 'IU_AFI')

#for i in "${arr[@]}"
for i in {0..0..1}
do
#    python compute_all_events.py $DIR_PROJECT $id observations syn_$i >> $DIR_PROJECT/job_tracker/logs/obs_${i}_${id}.txt 2>&1 &
    python compute_all_events.py $DIR_PROJECT $id misfit syn_$i >> $DIR_PROJECT/job_tracker/logs/misfit_${i}_${id}.txt 2>&1 &
#    python compute_all_events.py $DIR_PROJECT $id misfit_and_gradient syn_$i >> $DIR_PROJECT/job_tracker/logs/gradient_${i}_${id}.txt 2>&1 &


#    python compute_all_events.py $DIR_PROJECT $id misfit autocorrelation_homog_$i >> $DIR_PROJECT/job_tracker/logs/misfit_${i}_${id}.txt 2>&1 &
#    python compute_all_events.py $DIR_PROJECT $id misfit autocorrelation_layered_$i >> $DIR_PROJECT/job_tracker/logs/misfit_${i}_${id}.txt 2>&1 &

#    python compute_all_events.py $DIR_PROJECT $id misfit_and_gradient autocorrelation_homog_$i >> $DIR_PROJECT/job_tracker/logs/gradient_${i}_${id}.txt 2>&1 &
#    python compute_all_events.py $DIR_PROJECT $id misfit_and_gradient autocorrelation_layered_$i >> $DIR_PROJECT/job_tracker/logs/gradient_${i}_${id}.txt 2>&1 &

    sleep 0.5
done

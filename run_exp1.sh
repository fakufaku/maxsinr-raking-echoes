mkdir -p ./output
mkdir -p ./figures
mkdir -p ./results

EXP_ID=1
MAX_JOBS=${1:-4}    # Defaults to 4 if not provided.

job_count=0
for SIR in -5 0 5 10
do
for RT60 in 0.3 0.6
do
for BF in ds max_sinr mvdr
do
for MASK in oracle_vad oracle_scm
do
for NOISE_LOC in noise_doa speech_doa
do
if [ ${BF} == "ds" ] && [ ${MASK} != "oracle_vad" ]; then
continue
fi

# set +o xtrac
echo  $(jobs -r | wc -l)
python ./exp1_synthetic.py --SIR ${SIR} --RT60 ${RT60} --bf ${BF} --mask ${MASK} --noise_loc ${NOISE_LOC} &
# set -o xtrace
job_count=$((job_count + 1))
# If the number of background jobs reaches G, wait for them to finish
if (( job_count % MAX_JOBS == 0 )); then
    wait  # Wait for all background jobs to finish before proceeding
fi

done
done
done
done
done
done

echo "Experiment ${EXP_ID} completed."


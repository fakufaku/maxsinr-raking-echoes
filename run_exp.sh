mkdir -p ./output
mkdir -p ./figures
mkdir -p ./results

MAX_PROCS=${1:-4}    # Defaults to 4 if not provided.

# Function to wait until the number of running jobs is less than MAX_PROCS.
wait_for_slot() {  
  while [ "$(jobs -r | wc -l)" -ge "$MAX_PROCS" ]; do
        sleep 1
    done
}

for SIR in 5
do
for BF in ds maxsinr souden mvdr mpdr rake lcmv
do
for MASK in led oracle-ibm oracle-wiener
do
for SPEECH_COV in mix masked
do
if [ ${BF} == "ds" ] && [ ${MASK} != "led" ]; then
continue
fi
    wait_for_slot
    python ./experiment_different_bf_algos.py ${SIR} pyramic --vad_guard 1024 --bf ${BF} --mask ${MASK} --speech-cov ${SPEECH_COV} --save_sample ./output &
    job_count=$((job_count + 1))
    # If the number of background jobs reaches G, wait for them to finish

done
done
done
done
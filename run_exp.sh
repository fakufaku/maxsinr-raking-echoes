mkdir -p ./output
mkdir -p ./figures
mkdir -p ./results

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
python ./experiment_different_bf_algos.py ${SIR} pyramic --vad_guard 1024 --bf ${BF} --mask ${MASK} --speech-cov ${SPEECH_COV} --save_sample ./output
done
done
done
done
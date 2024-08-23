# Does MaxSINR rake echoes?

This is a simple question we want to answer.

## Quick Start

```bash
# 1. place and untar the archive in the same folder
tar xzfv otohikari_robin_measurements_20171207.tar.gz

# 2. run the following script
# arguments are <SNR> and <ARRAY>, e.g.,
python ./experiment_max_sinr.py 5 pyramic --plot
```

## Run with different beamformers and mask
```bash
 python ./experiment_different_bf_algos.py 5 pyramic --vad_guard 1024 --plot --bf souden_mvdr --mask led
```

## Type of Beamformers

- [x] MaxSIR (legacy)
- [x] Delay and sum (legacy)
- [x] MVDR (legacy)
- [ ] Robust MVDR (legacy)
- [x] Souden MVDR (legacy)
- [x] LCMV (legacy)
- [x] Rake BF (pyroomacoustics)

Mask estimators
- [x] led
- [x] oracle
- [ ] CACGMM mask-estimator
- [ ] Neural BF


To install **CACGMM**
```bash
# 1. dowload the repo
git clone https://github.com/desh2608/cacgmm.git
cd cacgmm

# 2. replace all the cupy with numpy (if you run on a non-gpu machine)
# 'import cupy as cp' --> 'import numpy as cp'

# 3. install the lib locally in editable version
pip install -e .
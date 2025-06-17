import numpy as np
from epileptor.python import classes
import time as timemeasure
from datetime import timedelta
from pathlib import Path

import json

# network parameters in a separated json file
with open("simulation_params.json", "r") as f:
    params = json.load(f)

# ----- TIME REFERENCE -----
start = timemeasure.time()

np.random.seed(int(params.get("simulation", {}).get("seed", None)))

# epileptor parameters
t_tot = int(params.get("simulation", {}).get("t_tot", None))    # ms
epi_args = params

duration = int(params.get("simulation", {}).get("duration", None))      # ms
dt = epi_args.get("simulation", {}).get("dt", None)
sim_steps = int(duration/dt)

time = np.arange(0, t_tot, duration)
time_sim = np.arange(0, t_tot, epi_args.get("simulation", {}).get("dt", None))

inpath = f'parameters'
x0 = np.load(f'{inpath}/x0_real.npy')[:time.size]
CpES = np.load(f'{inpath}/cp_real.npy')[:time.size]

x0_sim = np.interp(time_sim, time, x0)
CpES_sim = np.interp(time_sim, time, CpES)

# instantiate epileptor model
epinet = classes.SeizureNetwork(epi_args)
epinet.initialize_networks()

##### SET INPUT SPIKETIMES #####
x1_i, x2_i, evs_i = epinet.advance_simulation(t_stop=t_tot, x0_variable=x0_sim, CpES_variable=CpES_sim)
x1_i_m = x1_i.mean(axis=0)
x2_i_m = x2_i.mean(axis=0)
input_signal = 0.8*x1_i_m + 0.2*x2_i_m

# sample at given sampling frequency
fs = int(epi_args.get("simulation", {}).get("fs", 500)) # Hz
downsample_factor = 1   # max(1, int(t_tot / dt / fs))
input_signal = input_signal[::downsample_factor]  # Downsample to match the desired sampling frequency

# save results
expath = f'data/original'
Path(expath).mkdir(parents=True, exist_ok=True)
np.save(f'{expath}/LFP_signal.npy', input_signal)
np.save(f'{expath}/pop1_states.npy', x1_i)
np.save(f'{expath}/pop2_states.npy', x2_i)
# input_signal_norm = (input_signal - input_signal.min()) / (input_signal.max() - input_signal.min())
# np.save(f'{expath}/mean.npy', input_signal_norm)
# np.save(f'{expath}/st_ei.npy', evs_i)

# ----- TIME REFERENCE -----
end = timemeasure.time()
print(f'Code ended in {str(timedelta(seconds=(end-start)))}')

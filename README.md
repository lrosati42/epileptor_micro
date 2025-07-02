# epileptor_micro

**epileptor_micro** is a simulation framework for spiking neural networks (SNNs) designed to generate epileptic-like Average Membrane Potentials, optimized for execution on microcontrollers and embedded systems.

## Overview

This project implements a C version of the Epileptor model, simulating the activity of two interacting neuronal populations. It is inspired by the original Python implementation but is optimized for efficiency and portability on resource-constrained hardware.

- **Population 1:** Hindmarsh-Rose type neurons.
- **Population 2:** Morris-Lecar type neurons.
- **Synapses:** Supports both fast and slow synapses, including gap junctions and chemical coupling.
- **Output:** Computes and saves the mean of the main state variables (`x1`, `x2`) over time.


## Compilation

To compile the simulator, use:

```sh
gcc -std=c99 -o epileptor_sim epileptor_sim.c -lm
```

## Usage
To run the simulation:

```
./epileptor_sim
```

This will generate an output file (e.g., epileptor_output.txt) containing the time series of the mean state variables.

## Output Format
The output file contains tab-separated columns:
```
# t(ms)    mean_x1    mean_x2
0.00       ...        ...
0.05       ...        ...
...
```

- t(ms): Simulation time in milliseconds.
- mean_x1, mean_x2: Mean values of the main state variables for each population.

## Features
Efficient simulation of large neural populations on embedded hardware.
Configurable model parameters for different epileptiform dynamics.

## Requirements
Standard C99 compiler (e.g., gcc)
No external dependencies required for the C core.

## References
- Jirsa, V.K., Stacey, W.C., Quilichini, P.P., Ivanov, A.I., & Bernard, C. (2014). On the nature of seizure dynamics. Brain, 137(8), 2210–2230.
- Naze S, Bernard C, Jirsa V (2015) Computational Modeling of Seizure Dynamics Using Coupled Neuronal Networks: Factors Shaping Epileptiform Activity. PLoS Comput Biol 11(5): e1004209. https://doi.org/10.1371/journal.pcbi.1004209

[Original implementation (Python)](https://sebnaze@bitbucket.org/sebnaze/epilepton.git)

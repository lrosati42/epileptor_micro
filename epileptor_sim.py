import cffi
import numpy as np
from numpy.random import uniform
from tqdm import trange
import time as timemeasure
from datetime import timedelta
import json

# network parameters in a separated json file
with open("simulation_params.json", "r") as f:
    params = json.load(f)

def epinet_init():
    """
    Initializes the Epileptor simulation by setting up the C Foreign Function Interface (FFI)
    and loading the shared library containing the C implementation.
    Returns:
        tuple: A tuple containing:
            - clib: The loaded shared library object for the Epileptor simulation.
            - ffi: The FFI instance used to interact with the shared library.
    Notes:
        - The shared library file `epileptor_sim.so` must be located in the same directory
          as the script or provide the correct relative path.
        - The C function `epileptor_sim` is expected to have the following signature:
          `void epileptor_sim(double x0, double CpES, double *output, double duration, double dt, int seed);`
    """
    # Create a new instance of the C Foreign Function Interface (FFI)
    ffi = cffi.FFI()

    # Load the shared library containing the C implementation of the epileptor simulation
    clib = ffi.dlopen("./epileptor/C/epileptor_sim.so")

    # Define the C function signature for the epileptor simulation
    ffi.cdef("void epileptor_sim(double x0, double CpES, double *output, double duration, double dt, int seed, double *x1, double *y1, double *z, double *x2, double *y2);")

    # Return the loaded library and the FFI instance
    return clib, ffi

def params_loader(path):
    """
    Loads parameter files from the specified directory.

    This function loads two NumPy arrays from the given directory path:
    - 'x0_real.npy': Represents the initial conditions or parameters (x0).
    - 'cp_real.npy': Represents the coupling parameters (CpES).

    Args:
        path (str): The directory path where the parameter files are located.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - x0 (numpy.ndarray): The array loaded from 'x0_real.npy'.
            - CpES (numpy.ndarray): The array loaded from 'cp_real.npy'.

    Raises:
        FileNotFoundError: If the specified files are not found in the directory.
        ValueError: If the files cannot be loaded as NumPy arrays.
    """
    parpath = path
    x0 = np.load(parpath + '/x0_real.npy')
    CpES = np.load(parpath + '/cp_real.npy')
    return x0, CpES

if __name__ == "__main__":
    # ----- TIME REFERENCE -----
    start = timemeasure.time()

    # Initialize the Epileptor simulation by loading the shared library and FFI
    clib, ffi = epinet_init()

    # Parse command-line arguments for simulation parameters
    args = params
    seed = int(args.get("simulation", {}).get("seed", 42))  # Random seed for reproducibility
    np.random.seed(seed)  # Set the random seed

    # Load simulation parameters (x0 and CpES) from the 'parameters' directory
    x0_array, CpES_array = params_loader('parameters')
    # n_slices = len(x0_array)  # Number of parameter sets (slices)

    # Extract simulation duration and time step from arguments
    t_tot = int(args.get("simulation", {}).get("t_tot", None))
    duration = int(args.get("simulation", {}).get("duration", None))  # Total simulation duration in ms
    dt = args.get("simulation", {}).get("dt", 0.1)  # Simulation time step in ms
    sim_steps = int(duration/dt)  # Total number of simulation steps
    n_slices = int(t_tot / duration)  # Number of slices based on total time and duration

    # Pre-allocate an array to store the LFP (Local Field Potential) results for all slices
    total_LFP_array = np.zeros(n_slices * sim_steps, dtype=np.float64)

    # Initialize the pop1 state
    nbn1 = int(args.get("simulation", {}).get("nbn1", None))
    x1_min = float(args.get("pop1", {}).get("x1_min", None))
    x1_max = float(args.get("pop1", {}).get("x1_max", None))
    y1_min = float(args.get("pop1", {}).get("y1_min", None))
    y1_max = float(args.get("pop1", {}).get("y1_max", None))
    z_min = float(args.get("pop1", {}).get("z_min", None))
    z_max = float(args.get("pop1", {}).get("z_max", None))
    # pop1 state dictionary (initialized with random values)
    pop1_state = {
        "x1": uniform(x1_min, x1_max, size=nbn1),
        "y1": uniform(y1_min, y1_max, size=nbn1),
        "z": uniform(z_min, z_max, size=nbn1)
    }

    # Initialize the pop2 state
    nbn2 = int(args.get("simulation", {}).get("nbn2", None))
    x2_min = float(args.get("pop2", {}).get("x2_min", None))
    x2_max = float(args.get("pop2", {}).get("x2_max", None))
    y2_min = float(args.get("pop2", {}).get("y2_min", None))
    y2_max = float(args.get("pop2", {}).get("y2_max", None))
    # pop2 state dictionary (initialized with random values)
    pop2_state = {
        "x2": uniform(x2_min, x2_max, size=nbn2),
        "y2": uniform(y2_min, y2_max, size=nbn2)
    }

    states_1 = np.zeros((n_slices, nbn1))
    states_2 = np.zeros((n_slices, nbn2))

    # Loop through each parameter set (slice) and run the simulation
    for e in trange(n_slices):

        # Store the final states of pop1 and pop2 for each slice
        states_1[e, :] = pop1_state["x1"]
        states_2[e, :] = pop2_state["x2"]

        # Pre-allocate an array for the output of the C simulation for the current slice
        LFP_array = np.zeros(sim_steps, dtype=np.float64)

        # Convert the current x0 and CpES values to C-compatible pointers
        x0_ptr = ffi.cast("double", x0_array[e])
        CpES_ptr = ffi.cast("double", CpES_array[e])

        # Create a C-compatible pointer for the pop1 state variables
        x1 = ffi.cast("double *", pop1_state["x1"].ctypes.data)
        y1 = ffi.cast("double *", pop1_state["y1"].ctypes.data)
        z = ffi.cast("double *", pop1_state["z"].ctypes.data)

        # Create a C-compatible pointer for the pop2 state variables
        x2 = ffi.cast("double *", pop2_state["x2"].ctypes.data)
        y2 = ffi.cast("double *", pop2_state["y2"].ctypes.data)

        # Create a C-compatible pointer for the output array
        input_ptr = ffi.cast("double *", LFP_array.ctypes.data)

        # Call the C function to run the Epileptor simulation
        clib.epileptor_sim(x0_ptr, CpES_ptr, input_ptr, duration, dt, seed, x1, y1, z, x2, y2)

        # Store the simulation results in the pre-allocated total LFP array
        total_LFP_array[e * sim_steps:(e + 1) * sim_steps] = LFP_array

        seed += 1

    # sample at given sampling frequency
    fs = int(args.get("simulation", {}).get("fs", 500))
    downsample_factor = 1   # max(1, int(t_tot / dt / fs))
    total_LFP_array = total_LFP_array[::downsample_factor]  # Downsample to match the desired sampling frequency

    np.save('data/generated/LFP_signal_1s.npy', total_LFP_array)
    np.save('data/generated/pop1_states_1s.npy', states_1)
    np.save('data/generated/pop2_states_1s.npy', states_2)

    # ----- TIME REFERENCE -----
    end = timemeasure.time()
    print(f'Code ended in {str(timedelta(seconds=(end-start)))}')
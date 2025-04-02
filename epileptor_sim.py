import argparse
import cffi
import numpy as np
from tqdm import trange

def get_args():
    parser = argparse.ArgumentParser(description='Epileptor simulation')
    parser.add_argument('-T', '--duration', dest="duration", type=float, default=500.0, help='Duration of the simulation slices in ms')    # original slices were 500ms long
    parser.add_argument('-dt', '--dt', dest="dt", type=float, default=0.1, help='Simulation time step in ms')   # default 0.1 ms value equal to the original simulation
    parser.add_argument('-s', '--seed', dest="seed", type=int, default=42, help='Random seed for the simulation')
    args = parser.parse_args()
    return args

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
    clib = ffi.dlopen("./epileptor/epileptor_sim.so")

    # Define the C function signature for the epileptor simulation
    ffi.cdef("void epileptor_sim(double x0, double CpES, double *output, double duration, double dt, int seed);")

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
    # Initialize the Epileptor simulation by loading the shared library and FFI
    clib, ffi = epinet_init()

    # Parse command-line arguments for simulation parameters
    args = get_args()
    seed = args.seed  # Random seed for reproducibility

    # Load simulation parameters (x0 and CpES) from the 'parameters' directory
    x0_array, CpES_array = params_loader('parameters')
    # n_slices = len(x0_array)  # Number of parameter sets (slices)
    n_slices = 1000

    # Extract simulation duration and time step from arguments
    duration = args.duration  # Total simulation duration in ms
    dt = args.dt  # Simulation time step in ms
    sim_steps = int(duration / dt)  # Total number of simulation steps

    # Pre-allocate an array to store the LFP (Local Field Potential) results for all slices
    total_LFP_array = np.zeros(n_slices * sim_steps, dtype=np.float64)

    # Loop through each parameter set (slice) and run the simulation
    for e in trange(n_slices):
        # Pre-allocate an array for the output of the C simulation for the current slice
        LFP_array = np.zeros(sim_steps, dtype=np.float64)

        # Convert the current x0 and CpES values to C-compatible pointers
        x0_ptr = ffi.cast("double", x0_array[e])
        CpES_ptr = ffi.cast("double", CpES_array[e])

        # Create a C-compatible pointer for the output array
        input_ptr = ffi.cast("double *", LFP_array.ctypes.data)

        # Call the C function to run the Epileptor simulation
        clib.epileptor_sim(x0_ptr, CpES_ptr, input_ptr, duration, dt, seed)

        # Store the simulation results in the pre-allocated total LFP array
        total_LFP_array[e * sim_steps:(e + 1) * sim_steps] = LFP_array

    np.save('data/generated/LFP_signal.npy', total_LFP_array)
# Compiler and flags
CC = gcc
CFLAGS = -pedantic -Wall -Wextra -O2

# Target executable name
TARGET = epileptor_sim
SOURCE   = epileptor_sim.c
PLOT_PY  = plot_epileptor.py
OUT_FILE = epileptor_output.txt

# Default target: build the executable
all: $(TARGET) run_and_plot

# Link the target from the source file
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE)

# Run epileptor_sim and save output, then plot it
run_and_plot: $(TARGET)
	./$(TARGET)
	python3 $(PLOT_PY) $(OUT_FILE)

# Cleanup rule (remove build artifacts and plot/data if desired)
clean:
	rm -f $(TARGET) $(OUT_FILE) epileptor_plot.png


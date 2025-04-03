# Compiler and flags
CC = gcc
CFLAGS = -ansi -pedantic -Wall -Wextra -O2

# Target executable name
TARGET = epileptor_sim

# Default target: build the executable
all: $(TARGET)

# Link the target from the source file
$(TARGET): epileptor_sim.c
	$(CC) $(CFLAGS) -o $(TARGET) epileptor_sim.c -lm

# Clean up build artifacts
clean:
	rm -f $(TARGET)

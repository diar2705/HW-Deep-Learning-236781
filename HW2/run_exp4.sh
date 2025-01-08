#!/bin/bash

# exp 4
L41_values=(8 16 32)

for L in "${L41_values[@]}"; do
    # Construct the --run-name dynamically
    RUN_NAME="exp1_4_L${L}_K32"
    
    # Run the srun command
    srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -K 32 -L "$L" -P 4 -H 100 --run-name "$RUN_NAME" -M cnn
done

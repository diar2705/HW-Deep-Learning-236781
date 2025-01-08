#!/bin/bash

# exp 4

for L in {8,16}; do
    # Construct the --run-name dynamically
    RUN_NAME="exp1_4_L${L}_K32"
    
    # Run the srun command
    srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -K 32 -L "$L" -P 4 -H 100 --run-name "$RUN_NAME" -M cnn
done

for L in 32; do
    # Construct the --run-name dynamically
    RUN_NAME="exp1_4"
    
    # Run the srun command
    srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -K 32 -L "$L" -P 8 -H 100 --run-name "$RUN_NAME" -M cnn --early-stopping 10
done


for L in {2,4,8}; do
    # Construct the --run-name dynamically
    RUN_NAME="exp1_4"
    
    # Run the srun command
    srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -K 64 128 256 -L "$L" -P 8 -H 100 --run-name "$RUN_NAME" -M cnn --early-stopping 10
done


#!/bin/bash

# exp 1
K1_values=(32 64)
L1_values=(2 4 8)

# Loop through each combination of -K and -L
for K in "${K1_values[@]}"; do
  for L in "${L1_values[@]}"; do
    # Construct the --run-name dynamically
    RUN_NAME="exp1_1_L${L}_K${K}"
    
    # Run the srun command
    srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -K "$K" -L "$L" -P 2 -H 100 --run-name "$RUN_NAME" -M cnn
  done
done

for K in "${K1_values[@]}"; do
    RUN_NAME="exp1_1_L16_K${K}"
    
    # Run the srun command
    srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -K "$K" -L 16 -P 4 -H 100 --run-name "$RUN_NAME" -M cnn
  done
done


# exp 2
K2_values=(32 64 128)
L2_values=(2 4 8)

# Loop through each combination of -K and -L
for L in "${L2_values[@]}"; do
    for K in "${K2_values[@]}"; do
    # Construct the --run-name dynamically
    RUN_NAME="exp1_2_L${L}_K${K}"
    
    # Run the srun command
    srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -K "$K" -L "$L" -P 2 -H 100 --run-name "$RUN_NAME" -M cnn
  done
done


# exp 3
L3_values=(2 3 4)

# Loop through each combination of -K and -L
for L in "${L3_values[@]}"; do
    # Construct the --run-name dynamically
    RUN_NAME="exp1_3_L${L}_K64-128"
    
    # Run the srun command
    srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -K 64 128 -L "$L" -P 2 -H 100 --run-name "$RUN_NAME" -M cnn
done


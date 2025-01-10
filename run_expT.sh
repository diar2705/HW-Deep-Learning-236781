for L in {8,16}; do
    # Construct the --run-name dynamically
    RUN_NAME="exp1_4"
    
    # Run the srun command
    srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -K 32 -L "$L" -P 4 -H 100 --run-name "$RUN_NAME" -M resnet --early-stopping 10
done
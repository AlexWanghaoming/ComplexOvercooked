#!/bin/bash

# envs=(supereasy)
envs=(2playerhard)

algos=(iql vdn)

for e in "${envs[@]}"
do
   for algo in "${algos[@]}"
   do
       for i in {0..4}
       do
          python src/main.py --config=${algo} --env-config=overcooked2 with env_args.map_name=$e seed=$i hidden_dim=128 &
          echo "Running with ippo and $e for seed=$i"
          sleep 2s
       done
   done
done
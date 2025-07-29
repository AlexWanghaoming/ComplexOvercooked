#!/bin/bash

envs=(2playerhard)

algos=(vdn)

for e in "${envs[@]}"
do
   for algo in "${algos[@]}"
   do
       for i in {1}
       do
          python src/main.py --config=${algo} --env-config=overcooked2 with env_args.map_name=$e seed=7 hidden_dim=256 t_max=20000000 &
          echo "Running with ${algo} and $e for seed=$i"
          sleep 1s
       done
   done
done

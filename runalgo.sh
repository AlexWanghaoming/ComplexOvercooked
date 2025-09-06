#!/bin/bash

envs=(supereasy)

algos=(ippo)

for e in "${envs[@]}"
do
   for algo in "${algos[@]}"
   do
       for i in {1}
       do
          python src/main.py --config=${algo} --env-config=overcooked2 with env_args.map_name=${e} seed=7 t_max=10000000 use_cuda=True use_wandb=True
          echo "Running with ${algo} and ${e} for seed=${i}"
       done
   done
done

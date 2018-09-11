#!/usr/bin/env bash

'''
evaluation: bias and variance tradeoff for 4 different estimatos.
'''

set -e

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  python_v="/home/ys756/anaconda3/envs/fzc/bin/python"
else
  python_v="/Users/yisu/anaconda/envs/mltr/bin/python"
fi



run()
{
  if [[ "$OSTYPE" == "linux-gnu" ]]; then
    ~/submit_job.pl "$1"
  else
    $1
  fi
}


for sweep in 1; do
  for r in 1; do
    run "sh ./var_cal.sh $sweep $r"
  done
done

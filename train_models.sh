#!/usr/bin/env bash

'''
compare performance of 4 different estimators on multiple rankers
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


for sweep in 1 2 5 8 10; do
  for r in 1; do
    for i in $(seq 1 3); do
      run "sh ./train.sh $sweep $i $r"
    done
  done
done

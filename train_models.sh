#!/usr/bin/env bash
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


for train_size in 5000; do
  for r in 2; do
    for i in $(seq 1 1); do
      run "sh ./train.sh $train_size $i $r"
    done
  done
done

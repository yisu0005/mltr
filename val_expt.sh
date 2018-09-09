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


#for train_size in 2000 4000 6000 8000 10000 15000 20000 30000 50000 100000; do
for train_size in 2000; do
  for r in 2; do
    run "sh ./var_cal.sh $train_size $r"
  done
done

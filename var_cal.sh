#!/usr/bin/env bash
set -e

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  python="/home/ys756/anaconda3/envs/fzc/bin/python"
else
  python="/Users/yisu/anaconda/envs/mltr/bin/python"
fi

train_data='data/input/set1bin.train.txt'
val_data='data/input/set1bin.valid.txt'
test_data='data/input/set1bin.test.txt'

svm_learn='svm_rank/svm_rank_learn'
svm_classify='svm_rank/svm_rank_classify'
svmprop_learn='svm_proprank/svm_proprank_learn'
svmprop_classify='svm_proprank/svm_proprank_classify'

eta=0.8
eps_plus=1
eps_minus=0
train_size=$1
per_iter=5000
sweep=$((${train_size}%${per_iter}?${train_size}/${per_iter}+1:${train_size}/${per_iter}))
r=$2 # ranker type
ranker_dict='data/expt/var'/0.01_0.01_0
iter_dict=${ranker_dict}/${eta}_${eps_minus}_${train_size}/${r}
res_dict=${iter_dict}/'lastone'


${python} -m src.util ${train_data} ${iter_dict} -n 'rankerA_query' 'rankerB_query' 'left_query' ⁠⁠\
                                              -pr 0.01 0.01 0
${svm_learn} -c 2 ${iter_dict}/rankerA_query.txt ${iter_dict}/modelA.dat > /dev/null
${svm_learn} -c 2 ${iter_dict}/rankerB_query.txt ${iter_dict}/modelB.dat > /dev/null
${svm_learn} -c 2 ${iter_dict}/left_query.txt ${iter_dict}/modelC.dat > /dev/null

${svm_classify} ${test_data} ${iter_dict}/modelA.dat ${iter_dict}/predictionsA.txt
${svm_classify} ${test_data} ${iter_dict}/modelB.dat ${iter_dict}/predictionsB.txt
${svm_classify} ${test_data} ${iter_dict}/modelC.dat ${iter_dict}/predictionsC.txt
${python} -m src.test_perf ${test_data} ${iter_dict}/predictionsC.txt >> ${iter_dict}/testperf.txt

for i in $(seq 1 30); do
  echo "train_size: ${train_size}"
  echo "iter: ${i}"
	${python} -m src.click_log ${test_data} ${iter_dict}/predictionsA.txt ${iter_dict}/predictionsB.txt \
	                       ${r} ${iter_dict} prop_file.txt bal_prop_file.txt clip_prop_file.txt clipbal_prop_file.txt -e ${eta} \
	                       -p ${eps_plus} -m ${eps_minus} -a ${train_size} -b ${train_size} -s ${sweep}

	${svm_classify} ${res_dict}/prop_file.txt ${iter_dict}/modelC.dat ${iter_dict}/predictionsCp.txt

	${python} -m src.cv ${res_dict}/prop_file.txt ${iter_dict}/predictionsCp.txt >> ${iter_dict}/naive.txt
	${python} -m src.cv ${res_dict}/bal_prop_file.txt ${iter_dict}/predictionsCp.txt >> ${iter_dict}/balance.txt
	${python} -m src.cv ${res_dict}/clip_prop_file.txt ${iter_dict}/predictionsCp.txt >> ${iter_dict}/clip.txt
	${python} -m src.cv ${res_dict}/clipbal_prop_file.txt ${iter_dict}/predictionsCp.txt >> ${iter_dict}/clipbal.txt
done

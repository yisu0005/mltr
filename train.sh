#!/usr/bin/env bash
set -e
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  python="/home/ys756/anaconda3/envs/fzc/bin/python"
else
  python="/Users/yisu/anaconda/envs/mltr/bin/python"
fi

train_size=$1
i=$2
r=$3

rankerA=0.01
rankerB=0.01
overlap=0
eta=0.8
eps_plus=1.0
eps_minus=0.1

# declare -a rankerB_way=("all" "lastone" "portion")
declare -a rankerB_way=("lastone")
train_data='data/input/set1bin.train.txt'
val_data='data/input/set1bin.valid.txt'
test_data='data/input/set1bin.test.txt'

declare -i val_size
val_size=${train_size}/5
ranker_dict='data/expt'/${rankerA}_${rankerB}_${overlap}
iter_dict=${ranker_dict}/${eta}_${eps_minus}_${train_size}/${r}/${i}
output_dict=${iter_dict}

testfile_name='set1bin.newtest.txt'
prop_file='prop_file.txt'
bal_prop_file='bal_prop_file.txt'
clip_prop_file='clip_prop_file.txt'
clipbal_prop_file='clipbal_prop_file.txt'

val_prop_file='val_prop_file.txt'
val_bal_prop_file='val_bal_prop_file.txt'
val_clip_prop_file='val_clip_prop_file.txt'
val_clipbal_prop_file='val_clipbal_prop_file'

prop_model='prop_model'
bal_prop_model='bal_prop_model'
clip_prop_model='clip_prop_model'
clipbal_prop_model='clipbal_prop_model'

prop_pred='prop_pred.txt'
bal_prop_pred='bal_prop_pred.txt'
clip_prop_pred='clip_prop_pred.txt'
clipbal_prop_pred='clipbal_prop_pred.txt'

svm_learn='svm_rank/svm_rank_learn'
svm_classify='svm_rank/svm_rank_classify'
svmprop_learn='svm_proprank/svm_proprank_learn'
svmprop_classify='svm_proprank/svm_proprank_classify'


#${python} -m src.build_test ${test_data} ${output_dict} ${testfile_name}

${python} -m src.util ${train_data} ${iter_dict} -pr ${rankerA} ${rankerB} ${overlap}

${svm_learn} -c 2 ${iter_dict}/rankerA_query.txt ${iter_dict}/modelA.dat > /dev/null
# ${svm_learn} -c 200 ${iter_dict}/rankerB_query.txt ${iter_dict}/modelB.dat > /dev/null

${svm_classify} ${train_data} ${iter_dict}/modelA.dat ${iter_dict}/predictionsA.txt
# ${svm_classify} ${train_data} ${iter_dict}/modelB.dat ${iter_dict}/predictionsB.txt

${svm_classify} ${val_data} ${iter_dict}/modelA.dat ${iter_dict}/val_predictionsA.txt
# ${svm_classify} ${val_data} ${iter_dict}/modelB.dat ${iter_dict}/val_predictionsB.txt
${python} -m ltr.BuildRankerB ${iter_dict}/rankerA_query.txt ${train_data} ${val_data} ${iter_dict} predictionsB.txt val_predictionsB.txt

${python} -m src.click_log ${train_data} ${iter_dict}/predictionsA.txt ${iter_dict}/predictionsB.txt \
                       ${r} ${output_dict} ${prop_file} ${bal_prop_file} ${clip_prop_file} ${clipbal_prop_file} -e ${eta} \
                       -p ${eps_plus} -m ${eps_minus} -a ${train_size} -b ${train_size}

${python} -m src.click_log ${val_data} ${iter_dict}/val_predictionsA.txt ${iter_dict}/val_predictionsB.txt \
                       ${r} ${output_dict} ${val_prop_file} ${val_bal_prop_file} ${val_clip_prop_file} ${val_clipbal_prop_file} -e ${eta} \
                       -p ${eps_plus} -m ${eps_minus} -a ${val_size} -b ${val_size}


for method in "${rankerB_way[@]}"
  do
    res_dict=${output_dict}/${method}
    c_list="0.001 0.01 0.1 1.0 10.0"
    for c in ${c_list}
    do
       ${svmprop_learn} -c ${c} ${res_dict}/${prop_file} ${res_dict}/${prop_model}_${c}.dat > /dev/null
       ${svmprop_learn} -c ${c} ${res_dict}/${bal_prop_file} ${res_dict}/${bal_prop_model}_${c}.dat > /dev/null
       ${svmprop_learn} -c ${c} ${res_dict}/${clip_prop_file} ${res_dict}/${clip_prop_model}_${c}.dat > /dev/null
       ${svmprop_learn} -c ${c} ${res_dict}/${clipbal_prop_file} ${res_dict}/${clipbal_prop_model}_${c}.dat > /dev/null

       ${svmprop_classify} ${res_dict}/${val_prop_file} ${res_dict}/${prop_model}_${c}.dat ${res_dict}/val_pred_${c}.txt
       ${svmprop_classify} ${res_dict}/${val_prop_file} ${res_dict}/${bal_prop_model}_${c}.dat ${res_dict}/bal_val_pred_${c}.txt
       ${svmprop_classify} ${res_dict}/${val_prop_file} ${res_dict}/${clip_prop_model}_${c}.dat ${res_dict}/clip_val_pred_${c}.txt
       ${svmprop_classify} ${res_dict}/${val_prop_file} ${res_dict}/${clipbal_prop_model}_${c}.dat ${res_dict}/clipbal_val_pred_${c}.txt

       ${python} -m src.cv ${res_dict}/${val_prop_file} ${res_dict}/val_pred_${c}.txt -c ${c} >> ${res_dict}/naive.txt
       ${python} -m src.cv ${res_dict}/${val_prop_file} ${res_dict}/bal_val_pred_${c}.txt -c ${c} >> ${res_dict}/balance.txt
       ${python} -m src.cv ${res_dict}/${val_prop_file} ${res_dict}/clip_val_pred_${c}.txt -c ${c} >> ${res_dict}/clip.txt
       ${python} -m src.cv ${res_dict}/${val_prop_file} ${res_dict}/clipbal_val_pred_${c}.txt -c ${c} >> ${res_dict}/clipbal.txt
    done
         naive_c=$(${python} -m src.select_cv ${res_dict}/naive.txt)
         bal_c=$(${python} -m src.select_cv ${res_dict}/balance.txt)
         clip_c=$(${python} -m src.select_cv ${res_dict}/clip.txt)
         clipbal_c=$(${python} -m src.select_cv ${res_dict}/clipbal.txt)

        ${svmprop_classify} data/input/${testfile_name} ${res_dict}/${prop_model}_${naive_c}.dat ${res_dict}/${prop_pred} \
                                           > ${res_dict}/naive_result_${naive_c}.txt
        ${svmprop_classify} data/input/${testfile_name} ${res_dict}/${bal_prop_model}_${bal_c}.dat ${res_dict}/${bal_prop_pred} \
                                           > ${res_dict}/bal_result_${bal_c}.txt
        ${svmprop_classify} data/input/${testfile_name} ${res_dict}/${clip_prop_model}_${clip_c}.dat ${res_dict}/${clip_prop_pred} \
                                           > ${res_dict}/clip_result_${clip_c}.txt
        ${svmprop_classify} data/input/${testfile_name} ${res_dict}/${clipbal_prop_model}_${clipbal_c}.dat ${res_dict}/${clipbal_prop_pred} \
                                           > ${res_dict}/clipbal_result_${clipbal_c}.txt
  done

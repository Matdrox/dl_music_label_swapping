#!/usr/bin/env bash
abs_path=$(pwd)
# Path to config file prepared to evaluate according to this script
config_file='config/cifar10/eval_all.yaml'
# Name of trained config to evaluate
config=test_cifar10_imb001_stage2_mislas
# Run the eval script for all found trained models
model_dirs=$(ls saved | grep ^$config)
for model in $model_dirs
do
    # Change which model to eval (modify config file)
    sed -i '$ d' ${config_file}
    echo "resume: '${abs_path}/saved/$model/ckps/current.pth.tar'" >> ${config_file}
    # Run eval script
    python3 eval.py --cfg  config/cifar10/eval_all.yaml
done

# Remove list of logs (if such exists)
eval_logs='eval_logs.txt'
rm $eval_logs
# Create list of eval logs and calculate mean values
touch $eval_logs
eval_dirs=$(ls saved | grep ^eval_all)
for log in $eval_dirs
do
    echo $log >> $eval_logs
done
python3 calc_eval.py $eval_logs
rm $eval_logs


# $eval_out
#python3 eval.py --cfg  config/cifar10/eval_all.yaml | read eval_out
# test=$(python3 eval.py --cfg  config/cifar10/eval_all.yaml)
# Feature Distribution Analysis-Based Test Selection for Deep Learning Enhancement

This is the homepage of FATS including tool implementation,  studied DNN models , the simulated test sets and experiment results.

## Environment configuration
Before running FATS, please make sure you have installed various related packages, including keras, tensorflow, cleverhands and opencv.

## Simulated test sets and pre-trained models:
Part of the test sets lies in the folder 'dataset'. Because some of our simulated test sets exceed GitHub's file size limit of 100.00 MB, we can only upload part of our test sets in file '/mnist/adv' and '/cifar10/adv'. 

The operational details of simulated test sets are listed in the [README](https://github.com/smile-ya/FATS/tree/main/dataset/README.md) under the folder 'dataset'.

The pre-trained models are put in the folder 'model'.

## Main experimental codes
You can easily implement our method and 5 baseline methods by yourself or modifying this code.

The main experimental codes are 
mnist_retrain.py, 
cifar10_retrain.py, 
svhn_retrain.py. 
You can modify the list variables 'baselines'(methods) and 'operators'(simulated testset) to run what you prefer.

baselines =['FATS','MCP','DeepGini','LSA','DSA','SRS']

operators =['fgsm','jsma','bim-a','bim-b','cw-l2']


## Baseline methods

FATS. Our method is design in the "/FATS/FATS_method.py" and is implemented in the "***_retrain.py" as the function "new_select_from_mmd_Distance_test" and "learning_to_rank_test".

DeepGini. The method is implemented in the "***_retrain.py" as the function "select_deepgini".

MCP. The method is implemented in the "***_retrain.py" as the function "select_mcp".

LSA/DSA. You can directly invoke the functions "fetch_lsa" and "fetch_dsa" from "/LSA_DSA/sa.py" which is downloaded online from the paper "Guiding Deep Learning System Testing Using Surprise Adequacy". These functions can help you get the SA value of each input. The higher the value of SA is, the corresponding test case is more surprise to the DNN under testing. So we select the subset of test cases with higher corresponding SAs.

SRS. This method is implemented in the "***_retrain.py" as the function "select_rondom".

## Results 
We put the results for all experiments in AllResult.

"""
Supplementary program for final project in CS375 Spring 2023
This program loads models from model checkpoints (output of ModelBuilder.py) if they exist 
or train models otherwise. 
For each model, Platt Scaling and inverse sigmoid curve (novel method for specific data)
are applied and expected calibration errors (ECE) were calculated.  
"""

import csv
from datasets import load_dataset
import numpy as np
import random
import sklearn
from sklearn.calibration import calibration_curve, CalibrationDisplay
from transformers import AutoTokenizer, DistilBertModel, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate
import tensorflow as tf
import tensorflow_probability as tfp 
import torch
from torch import nn
import calibration as cal
import matplotlib.pyplot as plt

from math import log
from scipy.special import expit
from scipy.special import xlogy
from scipy.optimize import fmin_bfgs

def preprocess_function(examples):
    return tokenizer(examples["tweet"], truncation=True)

# Read in dataset and preprocessing 
train_data = load_dataset('csv', data_files = ['./data/train.csv'], split = 'train')
test_data = load_dataset('csv', data_files = ['./data/test.csv'], split = 'train')
val_data = load_dataset('csv', data_files = ['./data/val.csv'], split = 'train')
# Preprocess 
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# Tokenize tweet and truncate to be no longer than max length
tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)
tokenized_val = val_data.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Evaluate 
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

### Model with full training data 

id2label = {0: "FAKE", 1: "REAL"}
label2id = {"FAKE": 0, "REAL": 1}
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

def build_trainer(train, test, output_dir, resume_from_checkpoint):
    # define training hyperparameters 
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        resume_from_checkpoint = resume_from_checkpoint
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    return trainer

### Model fine-tuned on the full dataset: resume from model checkpoint

trainer = build_trainer(tokenized_train, tokenized_test, "modelA", "modelA/checkpoint-402")
trainer.train(resume_from_checkpoint = "./modelA/checkpoint-402")

# Get the ECE of the uncalibrated model 
output = trainer.predict(tokenized_test) # get the logits & predicted labels
# get true labels
y_true = np.array([test_data[i]["label"] for i in range(test_data.num_rows)])
# get the logits 
y_logits = output.predictions 
# transform np.array to tensor (so we can use softmax)
y_logits_torch = torch.from_numpy(y_logits) # logits in tensor form 
probabilities = tf.nn.softmax(y_logits_torch, axis=1)
# transform back to np.array for sklearn calibration functions 
y_prob = np.array(probabilities) # predicted probabilities 
y_prob_pos = np.array([y_prob[i][1] for i in range(y_prob.shape[0])]) # predicted probabilities for positive class 
# n_bins default is 5
prob_true, prob_pred = sklearn.calibration.calibration_curve(y_true, y_prob_pos, n_bins = 5)

# reliability diagram 
disp = CalibrationDisplay(prob_true, prob_pred, y_prob_pos)
disp.plot()

### Calibration 

y_logits_val = trainer.predict(tokenized_val).predictions # logits of val set 

# transform np.array to tensor (so we can use softmax)
y_logits_val_tf = torch.from_numpy(y_logits_val) # logits in tensor form
# get predicted probabilites by using softmax 
probabilities_val = tf.nn.softmax(y_logits_val_tf, axis=1)
# transform back to np.array 
y_prob_tmp = np.array(probabilities_val) # predicted probabilities - both classes
y_prob_val = np.array([y_prob_tmp[i][1] for i in range(y_prob_tmp.shape[0])]) # predicted probs for positive class (val data)
# true labels 
y_true_val = np.array([val_data[i]["label"] for i in range(val_data.num_rows)])

## Platt Scaling 

# Copied from sklearn.CalibrationClassifierCV source code
# refer to function fit in source code to see how this is used 

def _sigmoid_calibration(predictions, y, sample_weight=None):
    """
    Probability Calibration with sigmoid method (Platt 2000)
    predictions = (uncalibrated) predicted probabilities from classifier 
    y = true labels (0 or -1 for negative and 1 for positive)
    sample_weight = I just ignored this 
    """
#     predictions = column_or_1d(predictions)
#     y = column_or_1d(y)

    F = predictions  # F follows Platt's notations

    # Bayesian priors (see Platt end of section 2.2):
    # It corresponds to the number of samples, taking into account the
    # `sample_weight`.
    mask_negative_samples = y <= 0
    if sample_weight is not None:
        prior0 = (sample_weight[mask_negative_samples]).sum()
        prior1 = (sample_weight[~mask_negative_samples]).sum()
    else:
        prior0 = float(np.sum(mask_negative_samples)) #number of negative samples 
        prior1 = y.shape[0] - prior0 #number of positive samples 
    # The T are the t_i's in the paper 
    T = np.zeros_like(y, dtype=np.float64) # same shape as y, but all zeros
    T[y > 0] = (prior1 + 1.0) / (prior1 + 2.0)
    T[y <= 0] = 1.0 / (prior0 + 2.0)
    T1 = 1.0 - T

    def objective(AB):
        # From Platt (beginning of Section 2.2)
        P = expit(-(AB[0] * F + AB[1]))
        loss = -(xlogy(T, P) + xlogy(T1, 1.0 - P))
        if sample_weight is not None:
            return (sample_weight * loss).sum()
        else:
            return loss.sum()

    def grad(AB):
        # gradient of the objective function
        P = expit(-(AB[0] * F + AB[1]))
        TEP_minus_T1P = T - P
        if sample_weight is not None:
            TEP_minus_T1P *= sample_weight
        dA = np.dot(TEP_minus_T1P, F)
        dB = np.sum(TEP_minus_T1P)
        return np.array([dA, dB])

    AB0 = np.array([0.0, log((prior0 + 1.0) / (prior1 + 1.0))]) #initial values 
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    return AB_[0], AB_[1]

a, b = _sigmoid_calibration(y_prob_val, y_true_val) # a, b got from logistic regression on val set 
platt_preds = expit(-(a * y_prob_pos + b)) # apply platt scaling

# Reliability diagram of calibrated (Platt Scaling) model 
# y_true: actual labels 
# platt_preds: predicted probabilities for pos class 
prob_true_cal, prob_pred_cal = sklearn.calibration.calibration_curve(y_true, platt_preds, n_bins = 10)
disp_cal = CalibrationDisplay(prob_true_cal, prob_pred_cal, platt_preds)
disp_cal.plot()

## Novel method: Inverse Sigmoid Scaling 

a = -1 # any number you want as long as it's negative

# apply inverse of sigmoid
calibrated_preds = (np.log((1-y_prob_pos)/y_prob_pos))/a

# linear map to 0-1 range
min_cal = min(calibrated_preds)
max_cal = max(calibrated_preds)
range_cal = max_cal - min_cal

# calibrated probabilities 
inverse_sigmoid_preds = np.array([(elem - min_cal) / range_cal for elem in calibrated_preds])

# Reliability diagram of calibrated (novel method) model 
prob_true_cal, prob_pred_cal = sklearn.calibration.calibration_curve(y_true, inverse_sigmoid_preds, n_bins = 10)
disp_cal = CalibrationDisplay(prob_true_cal, prob_pred_cal, inverse_sigmoid_preds)
disp_cal.plot()

### Overall ECE comparison
calibration_error_un = cal.get_calibration_error(y_prob_pos, y_true)
calibration_error_platt = cal.get_calibration_error(platt_preds, y_true)
calibration_error_novel = cal.get_calibration_error(inverse_sigmoid_preds, y_true)

print("ECE of uncalibrated model is " + str(calibration_error_un))
print("ECE of model calibrated with Platt Scaling is " + str(calibration_error_platt))
print("ECE of model calibrated with inverse sigmoid curve (novel) is " + str(calibration_error_novel))
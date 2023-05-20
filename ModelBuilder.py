"""
Supplementary program for final project in CS375 Spring 2023
This program randomly sample 4 training dataset from the training data 
with 100, 400, 1600. 
For each size of training data, 4 models are produced by fine-tuning DistilBERT 
on each sample. 
Another model is produced by fine-tuning DistilBERT on the full training data (6420 entries).
Each model is saved in a separate checkpoint "model[SIZE]_[i]", where i goes from 1-4.  
For example, the 4 model checkpoints for model fine-tuned on 100-entried data are 
names model100_1, model100_2, model100_3, model100_4. 
Note that the model trained on the full dataset has checkpoint modelA/checkpoint-402. 
"""

import csv
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer, DistilBertModel, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate
import random
import torch

### Setting arguments for distilBERT 

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

trainer = build_trainer(tokenized_train, tokenized_test, "modelA", "modelA/checkpoint-402")

trainer.train(resume_from_checkpoint = "modelA/checkpoint-402")

trainer.evaluate(tokenized_test)

### Model with 4 randomly 100-entried sample 

accuracies= []

for i in range(4):
    train = tokenized_train.select(random.sample(range(tokenized_train.num_rows), k=100))
    trainer = build_trainer(train, tokenized_test, "model100_" + str(i + 1), False)

    trainer.train()
    
    accuracies.append(trainer.evaluate(tokenized_test))

average_accuracy_100 = sum([d['eval_accuracy'] for d in accuracies]) / len(accuracies)

### Model with 4 randomly 400-entried sample 

accuracies = []

for i in range(4):
    train = tokenized_train.select(random.sample(range(tokenized_train.num_rows), k=400))
    trainer = build_trainer(train, tokenized_test, "model400_" + str(i + 1), False)

    trainer.train()
    
    accuracies.append(trainer.evaluate(tokenized_test))

average_accuracy_400 = sum([d['eval_accuracy'] for d in accuracies]) / len(accuracies)

### Model with 4 randomly 1600-entried sample 

accuracies = []

for i in range(4):
    train = tokenized_train.select(random.sample(range(tokenized_train.num_rows), k=1600))
    trainer = build_trainer(train, tokenized_test, "model1600_" + str(i + 1), False)

    trainer.train()
    
    accuracies.append(trainer.evaluate(tokenized_test))

average_accuracy_1600 = sum([d['eval_accuracy'] for d in accuracies]) / len(accuracies)
average_accuracy_1600













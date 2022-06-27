from arguments import get_args

import pandas as pd
import json
import os
from turngpt_dataset import get_classifier_dataset
from tokenizer import DialogTokenizer
import transformers

import torch

from transformers import GPT2ForSequenceClassification

from transformers import Trainer, TrainingArguments

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(pred):
    """
        This function calculates the performance (accuracy, f1, precision and recall) of a method according to its prediction.
        pred: huggingface's prediction class
    """

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main(args):

    transformers.set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    tokenizer = DialogTokenizer(args.model)

    max_length = args.max_length
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    grad_accum = args.grad_accum

    train_dataset = get_classifier_dataset(args.train_data, tokenizer, max_length, balance=0.65) 
    dev_dataset = get_classifier_dataset(args.dev_data, tokenizer, max_length)

    model = GPT2ForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = 50256
    
    output_dir = args.output
    log_dir = "log_" + output_dir

    print(f'Creating {output_dir} and {log_dir} folders ')
    os.makedirs(log_dir, exist_ok = True)
    os.makedirs(output_dir, exist_ok = True)


    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_f1",
        logging_dir=log_dir,
        save_total_limit=10,
        load_best_model_at_end=True,
        logging_steps = 1000,
        do_train=True,
        do_eval=True,
        seed=args.seed,
        gradient_accumulation_steps = grad_accum,
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr        
    )

    print("Building Trainer ...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )
    print("Trainer built.")

    print("Start training ...")
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    main(args)
    global step



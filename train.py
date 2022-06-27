from arguments import get_args

import pandas as pd
import json
import os
from turngpt_dataset import get_dataset, get_datasetV2
from tokenizer import DialogTokenizer
import transformers

import torch

from model import GPT2LMHeadModel

from transformers import Trainer, TrainingArguments


def main(args):

    transformers.set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    tokenizer = DialogTokenizer(args.model)

    max_length = args.max_length
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    num_utt_per_dialog = args.num_utt_per_dialog

    train_dataset = get_dataset(args.train_data, tokenizer, max_length, num_utt_per_dialog)
    dev_dataset = get_dataset(args.dev_data, tokenizer, max_length, num_utt_per_dialog)

    model = GPT2LMHeadModel.from_pretrained(args.model)
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    model.set_weight_loss(len(tokenizer), tokenizer.eos_token_id, 
                             args.weight_regular_token, args.weight_eot_token)
    
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
        logging_dir=log_dir,
        save_total_limit=10,
        load_best_model_at_end=True,
        logging_steps = 5000,
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
    )
    print("Trainer built.")

    print("Start training ...")
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    main(args)
    global step



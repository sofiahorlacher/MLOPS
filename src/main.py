# main.py

import subprocess
import sys
import argparse
import os
import wandb
import torch
import lightning as L
from data_module import GLUEDataModule
from model import GLUETransformer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def run_experiment(args):
    # Set up WandB
    wandb_logger = WandbLogger(project=args.projectname, log_model=True)
    run_name = f"lr={args.learning_rate}_opt={args.optimizer}_wd={args.weight_decay}_bs={args.batch_size}_sched={args.scheduler}_warmup={args.warmup_steps}_msl={args.max_seq_length}"
    wandb_logger.experiment.name = run_name
    wandb_logger.log_hyperparams(vars(args))

    # Initialize data and model
    dm = GLUEDataModule(
        model_name_or_path=args.model_name_or_path,
        task_name=args.task_name,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
    )
    dm.setup("fit")

    model = GLUETransformer(
        model_name_or_path=args.model_name_or_path,
        num_labels=dm.num_labels,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.batch_size,
        optimizer=args.optimizer, 
        scheduler=args.scheduler
    )

    # Trainer with checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    # Start training
    trainer.fit(model, datamodule=dm)
    val_loss = trainer.callback_metrics["val_loss"].item()
    print(f"Validation Loss: {val_loss}")

    wandb.log({"val_loss": val_loss})
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Paraphrase Detection")
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased")
    parser.add_argument("--task_name", type=str, default="mrpc")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--scheduler", type=str, choices=["linear_warmup", "cosine"], default="linear_warmup")
    parser.add_argument("--projectname", type=str, default="test")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)

    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    run_experiment(args)

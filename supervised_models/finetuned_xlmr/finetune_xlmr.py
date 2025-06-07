import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Finetune XLM-RoBERTa for classification')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the training data CSV file')
    parser.add_argument('--output_dir', type=str, default='supervised_models/finetuned_xlmr/logs',
                        help='Directory to save the model and logs')
    parser.add_argument('--model_name', type=str, default='FacebookAI/xlm-roberta-large',
                        help='Pretrained model to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size per device')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='Evaluation batch size per device')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=6,
                        help='Number of training epochs')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Evaluation steps interval')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--val_size', type=float, default=0.025,
                        help='Validation set size (proportion)')
    return parser.parse_args()

def prepare_data(data_path, val_size, seed):
    """Load and prepare the dataset."""
    # Load training data
    training_data = pd.read_csv(data_path)
    
    # Filter necessary columns
    training_data = training_data[['UN_KEY', 'Text', 'Unfalltyp']]
    training_data.rename(columns={'Unfalltyp': 'label'}, inplace=True)
    
    # Data size
    print(f"Training data size: {len(training_data)}")
    
    # Train-validation split
    train_df, val_df = train_test_split(training_data, test_size=val_size, random_state=seed)
    
    # Display label distribution
    print("Train set label distribution:")
    print(train_df.label.value_counts())
    print("Train set label distribution (normalized):")
    print(train_df.label.value_counts(normalize=True))
    print("Validation set label distribution:")
    print(val_df.label.value_counts())
    
    # Convert to Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    return dataset

def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset and adjust labels."""
    # Tokenize text
    def tokenize_function(example):
        return tokenizer(example["Text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Adjust labels (shift from 1-7 to 0-6)
    def adjust_labels(batch_labels):
        return [label - 1 for label in batch_labels]
    
    tokenized_datasets["train"] = tokenized_datasets["train"].map(
        lambda batch: {"label": adjust_labels(batch["label"])}, 
        batched=True
    )
    tokenized_datasets["validation"] = tokenized_datasets["validation"].map(
        lambda batch: {"label": adjust_labels(batch["label"])}, 
        batched=True
    )
    
    # Remove unnecessary columns and set format for PyTorch
    tokenized_datasets = tokenized_datasets.remove_columns(["UN_KEY", "Text", "__index_level_0__"])
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    return tokenized_datasets

def compute_metrics(eval_preds):
    """Compute evaluation metrics."""
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    pred_labels = np.argmax(logits, axis=-1)
    f1 = f1_score(y_true=labels, y_pred=pred_labels, average="weighted")
    accuracy = accuracy_score(labels, pred_labels)
    return {"f1": f1, "accuracy": accuracy}

def plot_training_history(history, output_path):
    """Plot and save training and evaluation loss curves."""
    # Extract loss values
    train_losses = [log["loss"] for log in history if "loss" in log]
    eval_losses = [log["eval_loss"] for log in history if "eval_loss" in log]

    # Extract corresponding steps
    train_steps = [log["step"] for log in history if "loss" in log]
    eval_steps = [log["step"] for log in history if "eval_loss" in log]

    # Plot training & evaluation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label="Training Loss", marker="o")
    plt.plot(eval_steps, eval_losses, label="Evaluation Loss", marker="s")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()
    print(f"Training history plot saved to {output_path}")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    args = parse_arguments()
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Prepare datasets
    dataset = prepare_data(args.data_path, args.val_size, args.seed)
    tokenized_datasets = tokenize_dataset(dataset, tokenizer)
    
    # Define model
    num_labels = len(set(dataset['train']['label']))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    ).to(device)
    
    # Configure training arguments
    model_output_dir = os.path.join(args.output_dir, 
                                   f"xlm_roberta_large_finetuned_high_quality_labels")
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        weight_decay=1e-2,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,                     
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        fp16=True if device == "cuda" else False,
        dataloader_num_workers=2,
        report_to="tensorboard",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    
    # Save final model
    trainer.save_model(os.path.join(model_output_dir, "final_model"))
    
    # Plot and save training history
    history = trainer.state.log_history
    plot_path = os.path.join(model_output_dir, "training_history.png")
    plot_training_history(history, plot_path)
    
    # Final evaluation
    final_eval = trainer.evaluate()
    print("Final evaluation results:")
    print(final_eval)

if __name__ == "__main__":
    main()
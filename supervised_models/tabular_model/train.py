import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import argparse
import os
import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

from supervised_models.tabular_model.model import TabularModelNN
from supervised_models.tabular_model.dataset import TabularDataset

def training_loop(model, criterion, optimizer, scheduler, 
                  train_loader, val_loader, device, 
                  epochs, output_dir, log_file_path,
                  log_every_n_iterations=100):
    
    train_losses = []
    val_losses = []
    iteration_checkpoints = []
    best_val_weighted_f1 = 0
    prev_best_model_path = None
    total_iterations = 0
    
    # Log initial info
    with open(log_file_path, "a") as log_file:
        param_count = sum(p.numel() for p in model.parameters())
        log_file.write(f"Model Parameter Count: {param_count}\n\n")
        log_file.write("Iteration, Train Loss, Val Loss, Val Accuracy, Val Weighted F1\n")

    # Create epoch-level progress bar
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        # Training Phase
        model.train()
        running_train_loss = 0.0
        iterations_since_last_log = 0
        
        # Create batch-level progress bar
        batch_pbar = tqdm(train_loader, 
                         desc=f"Epoch {epoch+1}/{epochs}",
                         leave=False)  # Don't leave trace of inner bar
        
        for batch_idx, (tabular_data, targets) in enumerate(batch_pbar):
            tabular_data = tabular_data.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(tabular_data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            iterations_since_last_log += 1
            total_iterations += 1
            
            # Update batch progress bar description with current loss
            batch_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iteration': total_iterations
            })
            
            # Check if it's time to log and validate
            if total_iterations % log_every_n_iterations == 0:
                # Calculate average training loss
                avg_train_loss = running_train_loss / iterations_since_last_log
                train_losses.append(avg_train_loss)
                iteration_checkpoints.append(total_iterations)
                
                # Validation Phase
                model.eval()
                running_val_loss = 0.0
                all_targets = []
                all_predictions = []
                
                # Create validation progress bar
                val_pbar = tqdm(val_loader, 
                              desc=f"Validation (Iteration {total_iterations})",
                              leave=False)  # Don't leave trace of validation bar
                
                with torch.no_grad():
                    for val_tabular_data, val_targets in val_pbar:
                        val_tabular_data = val_tabular_data.to(device)
                        val_targets = val_targets.to(device)
                        
                        val_outputs = model(val_tabular_data)
                        val_loss = criterion(val_outputs, val_targets)
                        running_val_loss += val_loss.item()
                        
                        _, predicted = torch.max(val_outputs, 1)
                        all_targets.extend(val_targets.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())
                        
                        val_pbar.set_postfix({'val_loss': f"{val_loss.item():.4f}"})
                
                avg_val_loss = running_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                accuracy = accuracy_score(all_targets, all_predictions)
                weighted_f1 = f1_score(all_targets, all_predictions, average="weighted")
                
                # Update epoch progress bar with current metrics
                epoch_pbar.set_postfix({
                    'train_loss': f"{avg_train_loss:.4f}",
                    'val_loss': f"{avg_val_loss:.4f}",
                    'accuracy': f"{accuracy:.4f}"
                })
                
                # Log metrics
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"{total_iterations}, {avg_train_loss:.4f}, "
                                 f"{avg_val_loss:.4f}, {accuracy:.4f}, {weighted_f1:.4f}\n")
                
                # Save best model with best validation accuracy
                if weighted_f1 > best_val_weighted_f1:
                    best_val_weighted_f1 = weighted_f1
                    best_model_path = os.path.join(output_dir, f"best_model_iter_{total_iterations}.pth")
                    
                    if prev_best_model_path is not None and os.path.exists(prev_best_model_path):
                        os.remove(prev_best_model_path)
                    
                    torch.save(model, best_model_path)
                    prev_best_model_path = best_model_path
                
                # Update loss plot
                plt.figure(figsize=(10, 6))
                plt.plot(iteration_checkpoints, train_losses, label="Train Loss", marker='o')
                plt.plot(iteration_checkpoints, val_losses, label="Val Loss", marker='o')
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.title("Train vs. Validation Loss")
                plt.legend()
                plt.grid(True)
                plot_path = os.path.join(output_dir, "train_val_loss.png")
                plt.savefig(plot_path)
                plt.close()
                
                # Reset running loss and counter
                running_train_loss = 0.0
                iterations_since_last_log = 0
                
                # Switch back to training mode
                model.train()
        
        # Step the scheduler at the end of each epoch
        scheduler.step()

def main():
    parser = argparse.ArgumentParser(description="Train multimodal model")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train csv file")
    parser.add_argument("--train_val_test_split", type=list, default=[0.975, 0.025, 0.00])
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_schedular_step_size", type=int, default=3, help="Learning rate scheduler step size")
    parser.add_argument("--lr_schedular_gamma", type=float, default=0.75, help="Learning rate scheduler gamma")
    parser.add_argument("--num_resnet_blocks", type=int, default=4, help="Number of ResNet blocks")
    parser.add_argument("--hidden_dim", type=int, default=1280, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--log_every_n_iterations", type=int, default=50,
                        help="Number of iterations between logging and validation")    
    parser.add_argument("--output_dir", type=str, 
                        default="supervised_models/tabular_model/training_logs", 
                        help="Output directory")
    
    args = parser.parse_args()

    # Create an output directory with a timestamp.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    log_file_path = os.path.join(output_dir, "training_log.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write("Hyperparameters:\n")
        for arg, value in vars(args).items():
            log_file.write(f"{arg}: {value}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # Data Loading & Processing
    # --------------------
    df = pd.read_csv(args.train_csv)
    
    # Prepare tabular data.
    # Drop the columns that are not tabular features.
    tabular_data_matrix = torch.tensor(
        df.drop(columns=["UN_KEY", "Text", "Unfalltyp"]).values, 
        dtype=torch.float32
    )
    targets = torch.tensor(df["Unfalltyp"].values - 1, dtype=torch.long)

    print(f"Tabular Data Shape: {tabular_data_matrix.shape}")
    
    # --------------------
    # Train/Validation/Test Split
    # --------------------
    num_samples = len(targets)
    indices = torch.randperm(num_samples)
    tabular_data_matrix = tabular_data_matrix[indices]
    targets = targets[indices]
    
    train_samples = int(num_samples * args.train_val_test_split[0])
    val_samples = int(num_samples * args.train_val_test_split[1])
    # test_samples = num_samples - train_samples - val_samples  # if needed later

    train_dataset = TabularDataset(
        tabular_data=tabular_data_matrix[:train_samples],
        targets=targets[:train_samples]
    )
    val_dataset = TabularDataset(
        tabular_data=tabular_data_matrix[train_samples:train_samples+val_samples],
        targets=targets[train_samples:train_samples+val_samples]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # --------------------
    # Create and Initialize the Model
    # --------------------
    model = TabularModelNN(
        num_classes=len(torch.unique(targets)),
        tabular_dim=tabular_data_matrix.shape[1],
        hidden_dim=args.hidden_dim, 
        num_resnet_blocks=args.num_resnet_blocks, 
        dropout=args.dropout
    )
    # If multiple GPUs are available, wrap the model with DataParallel.
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=args.lr_schedular_step_size, 
                                                gamma=args.lr_schedular_gamma)

    training_loop(model, criterion, optimizer, scheduler, 
                  train_loader, val_loader, device, args.epochs, 
                  output_dir, log_file_path, args.log_every_n_iterations)
    

if __name__ == "__main__":
    main()

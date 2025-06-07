import torch
from torch.utils.data import DataLoader
import pandas as pd
import transformers
import argparse
import os
from tqdm import tqdm
from supervised_models.tabular_model.dataset import TabularDataset

def predict(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():  # Disable gradient calculations during inference
        for batch in tqdm(data_loader, desc="Predicting"):
            tabular_data = batch
            tabular_data = tabular_data.to(device)

            outputs = model(tabular_data)
            _, predicted_classes = torch.max(outputs, 1) # Get the index of the max log-probability
            predictions.extend(predicted_classes.cpu().numpy())
    return predictions

def create_inference_dataset(test_csv_path):
    df = pd.read_csv(test_csv_path)

    # Prepare tabular data. Drop the columns that are not tabular features.
    tabular_data_matrix = torch.tensor(
        df.drop(columns=["UN_KEY", "Text", "Unfalltyp"]).values,
        dtype=torch.float32
    )

    inference_dataset = TabularDataset(
        tabular_data=tabular_data_matrix
    )
    return inference_dataset, df["UN_KEY"].tolist() # Return UN_KEY for saving predictions

def main():
    parser = argparse.ArgumentParser(description="Inference script for tabular model")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save predictions CSV")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # Create Inference Dataset and DataLoader
    # --------------------
    inference_dataset, un_keys = create_inference_dataset(args.test_csv)
    inference_dataloader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False)

    # --------------------
    # Load Model from Checkpoint
    # --------------------
    model = torch.load(args.checkpoint_path, map_location=device)
    model.to(device)

    # --------------------
    # Make Predictions
    # --------------------
    predictions = predict(model, inference_dataloader, device)

    # --------------------
    # Save Predictions to CSV
    # --------------------
    predicted_unfalltyp = [pred + 1 for pred in predictions] # Add 1 to class indices to get original Unfalltyp values
    predictions_df = pd.DataFrame({
        "UN_KEY": un_keys,
        "tabular_model_prediction": predicted_unfalltyp
    })
    prediction_csv_path = os.path.join(args.output_dir, "tabular_model_trained_w_high_quality_test_predictions.csv")
    predictions_df.to_csv(prediction_csv_path, index=False)

    print(f"Predictions saved to {prediction_csv_path}")

if __name__ == "__main__":
    main()
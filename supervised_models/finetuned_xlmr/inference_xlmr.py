import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os

def run_inference(model_checkpoint_path, data, output_csv_path):
    # Automatically determine device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_path).to(device)
    
    # Load pipeline for inference with explicit device
    classifier = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=0 if device == "cuda" else -1  # 0 for GPU, -1 for CPU
    )

    predicted_labels = []
    predicted_probabilities = []    
    for text in tqdm(data['Text'], desc="Processing rows"):
        # Get the prediction and probability for each text
        prediction = classifier(text)[0]
        label = int(prediction['label'].split('_')[-1]) + 1  # Convert 0-6 to 1-7
        probability = prediction['score']  # Confidence score
        
        predicted_labels.append(label)
        predicted_probabilities.append(probability)
    
    # Add predictions to the DataFrame
    data['finetuned_xlmr_predictions'] = predicted_labels
    data['finetuned_xlmr_probabilities'] = predicted_probabilities
    # Drop other columns
    data = data[['UN_KEY', 'finetuned_xlmr_predictions', 'finetuned_xlmr_probabilities']].copy().reset_index(drop=True)

    # Save updated dataframe to the output path
    data.to_csv(output_csv_path, index=False)
    print(f"Inference completed. Results saved to {output_csv_path}.")

def main():
    parser = argparse.ArgumentParser(
        description="Inference with finetuned encoder only transformer model.")
    parser.add_argument("--model_checkpoint_dir", type=str, 
                        required=True, help="The dir that contains finetuned model")
    parser.add_argument("--unfalltext_csv_path", type=str, 
                        required=True, help="CSV file that contains 'Text' column to be labelled")    
    parser.add_argument("--test_set_csv_path", type=str, 
                        required=True, help="CSV file that contains 'Text' column to be labelled")
    parser.add_argument("--output_csv_dir", type=str, 
                        default="data", help="The directory to save outputs")
    args = parser.parse_args()
    
    # Output file path and name
    output_csv_path = os.path.join(args.output_csv_dir, "finetuned_xlmr_with_high_quality_labels_predictions.csv")
    
    # Inference data
    # Load data
    test_set = pd.read_csv(args.test_set_csv_path)
    data = pd.read_csv(args.unfalltext_csv_path)
    data = data[data['UN_KEY'].isin(test_set['UN_KEY'])].copy()
    data = data[['UN_KEY', 'Text']].copy()
    if "Text" not in data.columns:
        raise ValueError("Input CSV file must contain a 'Text' column.")
    
    run_inference(model_checkpoint_path=args.model_checkpoint_dir,
                  data=data,
                  output_csv_path=output_csv_path)
    

if __name__ == "__main__":
    main()
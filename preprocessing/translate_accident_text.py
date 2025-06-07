import argparse
import pandas as pd
from tqdm import tqdm
import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"

def translate_batch(texts, model, tokenizer, device):
    """Translates a batch of texts using the provided model and tokenizer."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(
        description="Translate accident text descriptions from German to English.")
    parser.add_argument("--unfalltext_csv_path", type=str, 
                        required=True, help="CSV file path")
    parser.add_argument("--start_row", type=int, required=True, 
                        help="The starting row number for the translation range")
    parser.add_argument("--end_row", type=int, required=True, 
                        help="The ending row number for the translation range")
    parser.add_argument("--output_dir", type=str, 
                        default="data", help="Output directory")
    parser.add_argument("--batch_size", type=int, 
                        default=128, help="Batch size for translation")
    parser.add_argument("--save_intermediate_epoch_count", type=int, default=100, 
                        help="Save results every X batches incase of failure")    
    args = parser.parse_args()

    # Load the CSV file into a pandas DataFrame
    df_unfalltext = pd.read_csv(args.unfalltext_csv_path)

    # Filter the rows based on the given range
    args.end_row = len(df_unfalltext) if args.end_row > len(df_unfalltext) else args.end_row
    print(f"Translating rows {args.start_row} to {args.end_row}")
    df_unfalltext = df_unfalltext.iloc[args.start_row:args.end_row]

    # Load the translation model and tokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)

    # Eval mode for faster inference
    model.eval()
    
    # Prepare for batch processing
    batch_size = args.batch_size
    num_batches = (len(df_unfalltext) + batch_size - 1) // batch_size

    # Translate in batches
    df_unfalltext["Translated_Text"] = None
    temp_file = None
    previous_temp_file = None
    for i in tqdm(range(num_batches), desc="Translating"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_unfalltext))  # Adjust for the last batch        
        batch_df = df_unfalltext.iloc[start_idx:end_idx].copy()
        texts = batch_df["Text"].tolist()
        translations = translate_batch(texts, model, tokenizer, device)

        # Update the original DataFrame
        df_unfalltext.loc[start_idx:end_idx-1, "Translated_Text"] = translations

        if i > 0 and i % args.save_intermediate_epoch_count == 0:
            temp_file = f"{args.output_dir}/translation_temp_batch_{i}.csv"        
            df_unfalltext.to_csv(temp_file, index=False)
            # remove the previous temp file with different name
            previous_temp_file = f"{args.output_dir}/translation_temp_batch_{i-args.save_intermediate_epoch_count}.csv"
            # Remove the last temp file if it was created
            if previous_temp_file and os.path.exists(previous_temp_file):
                os.remove(previous_temp_file)
        
    # Save all translations to the DataFrame
    output_file = f"{args.output_dir}/translated_unfalltext_{args.start_row}_{args.end_row}.csv"
    df_unfalltext.to_csv(output_file, index=False)
    
    # remove the last temp file if it exists
    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)

    print(f"Translations saved to {output_file}")


if __name__ == "__main__":
    main()
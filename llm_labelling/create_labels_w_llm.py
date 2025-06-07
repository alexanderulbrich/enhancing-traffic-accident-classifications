import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import accelerate
import re
import os
from llm_labelling.prompts import (
    ZERO_SHOT_PROMPT, 
    FEW_SHOT_PROMPT, 
    ADJUSTED_FEW_SHOT_PROMPT, 
    DAMAGED_PARKED_VEHICLE_ANALYSIS_PROMPT)

LLM_NAME = "google/gemma-2-27b-it"
DTYPE = torch.bfloat16
MAX_LENGTH = 8192

# Define a function to extract the class number from the JSON
def extract_class_from_json(text):
    # Use regular expression to find the JSON-like structure in the generated text
    json_match = re.search(r'\{.*"typ":\s*(\d+)\s*.*\}', text)
    if json_match:
        # Extract and return the class value
        return int(json_match.group(1))
    else:
        # Return 99 for unmatched cases
        return 99

def label_batch(texts, model, tokenizer, device, prompt_name="few_shot"):
    """
    Predicts the label for a batch of texts using the provided model and tokenizer.
    :param texts: List of text strings to label
    :param model: The language model to use for inference
    :param tokenizer: The tokenizer to use for tokenizing the input text
    :param device: The device to run the inference on
    """
    # Prepare the dynamic prompts for each text
    if prompt_name == "zero_shot":
        prompt = ZERO_SHOT_PROMPT
    elif prompt_name == "few_shot":
        prompt = ADJUSTED_FEW_SHOT_PROMPT
    elif prompt_name == "damaged_park_prompt":
        prompt = DAMAGED_PARKED_VEHICLE_ANALYSIS_PROMPT
    
    dynamic_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{prompt}\n{text}\nAntwort:"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for text in texts
    ]
    # Tokenize the dynamic prompts
    inputs = tokenizer(dynamic_prompts, padding=True, truncation=True, 
                       max_length=MAX_LENGTH, return_tensors="pt").to(device)

    with torch.no_grad():
        # Greedy decoding since we expect only 1 number as output
        outputs = model.generate(
            **inputs,
            num_beams=1,        # Greedy decoding (beam search with 1 beam)
            do_sample=False,    # Disable sampling
            max_new_tokens=20,
            return_dict_in_generate=True,  # Return additional generation info
            output_scores=False  # Scores are not needed for this task
        )

    # Extract only the newly generated token IDs
    new_token_ids = outputs.sequences[:, inputs['input_ids'].shape[1]:]

    # Decode the new tokens to get the generated output
    predicted_labels = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

    # Clean the predictions to extract only the number
    predicted_labels = [pred.strip() for pred in predicted_labels]

    # Extract class values for all texts in the batch
    predicted_labels = [extract_class_from_json(label) for label in predicted_labels]

    return predicted_labels

def main():
    parser = argparse.ArgumentParser(
        description="Label accident text descriptions using a language model.")
    parser.add_argument("--hf_model_weight_cache_dir", type=str, 
                        required=True, help="The cache directory for the model weights from Hugging Face")
    parser.add_argument("--unfalltext_csv_path", type=str, 
                        required=True, help="CSV file path")
    parser.add_argument("--start_row", type=int, required=True, 
                        help="The starting row number for the labelling")
    parser.add_argument("--end_row", type=int, required=True, 
                        help="The ending row number for the labelling")
    parser.add_argument("--output_dir", type=str, 
                        default="data", help="Output directory")
    parser.add_argument("--batch_size", type=int, 
                        default=8, help="Batch size for inference")
    parser.add_argument("--save_intermediate_batch_count", type=int, default=300, 
                        help="Save results every X batches incase of failure")
    parser.add_argument("--prompt_name", type=str, choices=["zero_shot", "few_shot", "damaged_park_prompt"], 
                        default="few_shot", help="Choose 'zero_shot' for zero-shot or 'few_shot' for few-shot inference or \
                            'damaged_park_prompt' for the prompt for damaged park vehicle analysis")

    args = parser.parse_args()

    # Load the CSV file into a pandas DataFrame
    df_unfalltext = pd.read_csv(args.unfalltext_csv_path)

    # Filter the rows based on the given range
    args.end_row = len(df_unfalltext) if args.end_row > len(df_unfalltext) else args.end_row
    print(f"Labelling rows {args.start_row} to {args.end_row}")
    df_unfalltext = df_unfalltext.iloc[args.start_row:args.end_row].reset_index(drop=True)

    # Enable DataParallel for multi-GPU usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cpu":
        print("This code needs GPUs to run, otherwise it would take oo long!")
    
    # Download weights and save them to the cache folder
    cache_folder = f"{args.hf_model_weight_cache_dir}/{LLM_NAME}"
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        cache_dir=cache_folder,
        device_map="auto",
        torch_dtype=DTYPE
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, cache_dir=cache_folder)
    print(f"Model weights downloaded to or loaded from {cache_folder}")

    # Eval mode for faster inference
    model.eval()
    
    # Prepare for batch processing
    batch_size = args.batch_size
    num_batches = (len(df_unfalltext) + batch_size - 1) // batch_size
    
    # Create the column for the predicted labels
    df_unfalltext[f"{args.prompt_name}_LLM_Labels"] = None
    for i in tqdm(range(num_batches), desc="Processing batches..."):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_unfalltext))  # Adjust for the last batch
        batch_df = df_unfalltext.iloc[start_idx:end_idx].copy()
        texts = batch_df["Text"].tolist()
        labels = label_batch(texts, model, tokenizer, device, args.prompt_name)
        
        # Update the original DataFrame
        df_unfalltext.loc[start_idx:end_idx-1, f"{args.prompt_name}_LLM_Labels"] = labels
        
        if i > 0 and i % args.save_intermediate_batch_count == 0:
            temp_file = f"{args.output_dir}/llm_labelling_temp_batch_{i}.csv"        
            df_unfalltext.to_csv(temp_file, index=False)
            # remove the previous temp file with different name
            previous_temp_file = f"{args.output_dir}/llm_labelling_temp_batch_{i-args.save_intermediate_batch_count}.csv"
            if os.path.exists(previous_temp_file):
                os.remove(previous_temp_file)

    # Save all labels to the DataFrame
    output_file = f"{args.output_dir}/{args.prompt_name}_llm_predicted_labels_unfalltext_{args.start_row}_{args.end_row}.csv"
    df_unfalltext.to_csv(output_file, index=False)
    
    # remove the last temp file if it exists
    if 'temp_file' in locals():
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"LLM inference finished, labels are saved to {output_file}")

if __name__ == "__main__":
    main()
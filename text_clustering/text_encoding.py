import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import argparse
from tqdm import tqdm
import os
import numpy as np
import json
import pickle

# specify model which speaks proper german
MODEL_NAME = "jinaai/jina-embeddings-v3"

def main():
    parser = argparse.ArgumentParser(description="Encode accident texts")

    # define arguments
    parser.add_argument("--path", type=str, required=True, help="path from where to load the csv")
    parser.add_argument("--n_of_texts", type=int, default=50000, help="Nr of texts to encode")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for encoding")
    parser.add_argument("--output_type", type=str, default="matrix", help="Select output type. By default it's a matrix which is used by the clusterin script. For the finetuning we will need output_type dictionary")

    # put arguments together in args
    args = parser.parse_args()    
    # load csv from via argument defined csv-path
    df = pd.read_csv(args.path)

    # shrink dataset to specified amount of texts
    df = df[:args.n_of_texts]

    # specify gpu as parallelization option, otherwise use cpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load model and tokenizer
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device)

    # Eval mode for faster inference
    model.eval()

    # take specified batch size
    batch_size = args.batch_size
    num_batches = (len(df) + batch_size - 1) // batch_size

    output_mode = args.output_type

    if output_mode == "matrix" :
        # create list for encodings
        encoded_texts = []
        for i in tqdm(range(num_batches), desc="Encoding"):
            # select chunks of data for batchwise encoding
            batch_df = df.iloc[i * batch_size: (i + 1) * batch_size]
            texts = batch_df["Text"].tolist()
            # no gradients saved since we don't need backprop
            with torch.no_grad():
                embeddings = model.encode(texts, task="separation", max_length=2048)
            
            # sentence embeddings is now a collapsed (via mean pooling) version of outputs, so it's a matrix of size num_texts x max_tokens
            encoded_texts.append(embeddings)

        # save results
        embedding_matrix = np.vstack(encoded_texts)
        embedding_matrix = embedding_matrix.reshape(50000, 1024)
        embedding_matrix_pandas = pd.DataFrame(embedding_matrix)
        output_file = f"{args.output_dir}/encoded_unfalltext_first{args.n_of_texts}.csv"
        embedding_matrix_pandas.to_csv(output_file, index=False)
        print(f"Encodings saved as matrix to {output_file}")
        
    else:
        embeddings_dict = {}
        for i in tqdm(range(num_batches), desc="Encoding"):
            batch_df = df.iloc[i * batch_size: (i + 1) * batch_size]
            texts = batch_df["Text"].tolist()
            keys = batch_df["UN_KEY"].tolist()
            with torch.no_grad():
                embeddings = model.encode(texts, task="classification", max_length=2048, truncate_dim=256)
            for key, embedding in zip(keys, embeddings):
                embeddings_dict[key] = embedding

        output_file = f"{args.output_dir}/encoded_text_first{args.n_of_texts}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(embeddings_dict, f)
        print(f"Encodings saved as dictionary to {output_file}")
                
if __name__ == "__main__":
    main()
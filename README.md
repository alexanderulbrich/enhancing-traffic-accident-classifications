# munich-accident-analysis

This repository is developed for the "Enhancing Munich Traffic Accident Classifications" consulting project at Ludwig-Maximilians-Universität.
- Project Responsibles:       Enes Özeren, Alexander Ulbrich
- Project Partner:            Dr. Sascha Filimon
- Project Advisors:           Prof. David Rügamer, Dr. Andreas Bender

This project resulted in a paper accepted at ECML PKDD 2025 and presented in Porto, Portugal.

**Paper:** [Title of the Paper](https://arxiv.org/abs/2506.12092)

## Installment & File structure
The Python version used in this repo is `3.11.5`.
For python package requirements, check `requirements_python.txt` file.

Folder structure:
```
.
├── README.md                       
├── data                                        <- data directory (data files ignored)
├── explorative                                 <- descriptive analysis for tables
│   ├── conventions.py                          <- conventions for labels and colors
│   ├── explore_beteiligte_mitfahrer.ipynb      <- beteiligte & mitfahrer table analysis
│   └── explore_text_descriptions               <- unfalltyp and accident text descriptive analysis
├── hf_models_cache                             <- cache dir for huggingface model weights
├── llm_labelling                               <- labelling accidents with llms & comparison analysis
│   ├── comparison_analysis                     <- contains the comparison of llm and human labels
│   ├── create_labels_w_llm.py                  <- labelling accidents with llms (check below to run)
│   ├── damaged_parked_vehicle_anaylsis.ipynb   <- damaged parked vehicles detailed analysis
│   └── prompts.py                              <- contains prompts to be used in scripts
├── preprocessing                               <- preprocessing scripts
│   ├── convert_data.r                          <- converts the .image data to csv files
│   ├── test_data_prep.ipynb                    <- test data split for expert labelling
│   └── translate_accident_text.py              <- translates german accident text to english
├── supervised_models                           <- contains supervised models and their scripts
│   ├── finetuned_xlmr                          <- directory for finetuning xlmr and inference
│   ├── multimodal_model                        <- directory for multimodal model training, inference scripts 
│   ├── tabular_model                           <- directory for tabular model training, inference scripts
│   ├── create_tabular_features_w_text.py       <- script to create tabular features with text data
│   └── inference_result_analysis.py            <- model prediction performance analysis
├── text_clustering                             <- clustering analysis scripts and notebooks
│   ├── param_tuning.py                         <- exploring hps for topic clustering
│   ├── text_endcoding.py                       <- creates embeddings of accident texts
│   ├── tfidf_clustering.ipynb                  <- creates clusters with tfidf vectors
│   └── unsupervised_clustering.ipynb           <- creates clusters with embeddings and bertopic
├── requirements_python.txt                     <- python package requirements
└── .gitignore                                  <- git ignored files
```

## Data Preprocessing

The dataset is given in a .image file which contains multiple tables.

To convert the tables to .csv files use the script below:
```bash
Rscript preprocessing/convert_data.r
```

To create english translations of german accident text reports, first create the csv with the `preprocessing/convert_data.r` and then use the python script below:
```bash
python preprocessing/translate_accident_text.py \
--unfalltext_csv_path data/D_Unfalltext_image.csv \
--start_row 0 \
--end_row 10
```

## Few-shot LLM Labelling and Topic Modelling

### Few-shot LLM labelling

To be able to use the Gemma model in the following script follow the steps below:
1) Log in to your huggingface account in web and accept the gemma conditions
2) Go to your hugginface account settings and create a token
3) In your terminal write `huggingface-cli login` and give your token to log in to hugginface

After this steps you can run the following scripts.

To predict labels of german accident text descriptions, first create the csv with the `preprocessing/convert_data.r` and then use the python script below:
```bash
python llm_labelling/create_labels_w_llm.py \
--hf_model_weight_cache_dir /home/ra32qov/munich-accident-analysis/hf_models_cache/ \
--unfalltext_csv_path data/D_Unfalltext_image.csv \
--start_row 0 \
--end_row 16 \
--batch_size 8 \
--prompt_name few_shot \
--save_intermediate_batch_count 12
```

### Topic Modelling

#### Encoding Accident Texts

To take the predefined random subset of texts and create highdimensional embeddings use the code below:
```bash
python text_clustering/text_encoding.py \
--path data/D_Unfalltext_image_50000_subsample.csv \
--output_dir data \
--batch_size 2028 \
--output_type matrix
```

#### Run Hyperparameter Analysis

The following code will try out different configurations of Bertopics hyperparameters, specifically used in Dimensionality-Reduction (UMAP) and CLustering (HDBSCAN). Note this is not actually tuning but solely returning the respective parameter configuration and the coherence, diversity, number of topics and number of outliers. The parameter grid is specified in the script with exemplary values.
```bash
python text_clustering/param_tuning.py
```

## Supervised Modelling

First create the tabular features + text dataframe. Then supervised models will use the outputs (tabular only / tabular + text / text only) as they need it.

```bash
python supervised_models/create_tabular_features_w_text.py \
--unfall_table_csv_path data/D_Unfall_RData.csv \
--beteiligte_table_csv_path data/D_Beteiligte_RData.csv \
--unfalltext_csv_path data/D_Unfalltext_image.csv \
--llm_labels_csv_path data/few_shot_llm_predicted_labels_unfalltext_0_103111.csv \
--test_csv_path data/large_test_set.csv \
--duplicates_to_remove_excel_path data/duplicated_text_description_list.xlsx \
--output_dir data
```

### Fine-tuned XLM-RoBERTa

#### Fine-tuning

Run the finetuning script below.
```bash
python supervised_models/finetuned_xlmr/finetune_xlmr.py \
--data_path data/tabular_features_w_text_train.csv
```

#### Inference with finetuned XLM-R

You need the finetuned XLM-RoBERTa model checkpoint directory and a csv file which contains `Text` column in it.

To predict labels for those `Text` column, use the script below.
```bash
python supervised_models/finetuned_xlmr/inference_xlmr.py \
--model_checkpoint_dir supervised_models/finetuned_xlmr/logs/xlm_roberta_large_finetuned_high_quality_labels/checkpoint-1400 \
--unfalltext_csv_path data/D_Unfalltext_image.csv \
--test_set_csv_path data/large_test_set.csv
```

### Multimodal Model

Multimodal model uses both text data and tabular data. To use the text data, an embedding model is used to create embedding vector for each task.

#### Train the Multimodal Model

```bash
python supervised_models/multimodal_model/train.py \
--train_csv data/tabular_features_w_text_train.csv \
--epochs 5 \
--batch_size 64 \
--lr 0.00001 \
--lr_schedular_step_size 1 \
--lr_schedular_gamma 0.70 \
--num_resnet_blocks 10 \
--hidden_dim 1440 \
--dropout 0.2 \
--weight_decay 0.01
```

#### Inference with the Multimodal Model

After you train and save the model checkpoint you can use the following inference script.

```bash
python supervised_models/multimodal_model/inference.py \
--test_csv data/tabular_features_w_text_test.csv \
--checkpoint_path supervised_models/multimodal_model/training_logs/20250310_074512/best_model_iter_2350.pth
```

### Tabular Model

Before training the model with only tabular features, first run the `supervised_models/multimodal_model/create_tabular_features_w_text.py` script as described in previous section. Tabular model will ignore the text column and will use only tabular features.

#### Training Tabular Model

To train the tabular neural network model use:

```bash
python supervised_models/tabular_model/train.py \
--train_csv data/tabular_features_w_text_train_low_quality_labels.csv \
--epochs 50 \
--batch_size 512 \
--lr 0.001 \
--lr_schedular_step_size 5 \
--lr_schedular_gamma 0.9 \
--num_resnet_blocks 10 \
--hidden_dim 1440 \
--dropout 0.1 \
--weight_decay 0.01
```

#### Inference with tabular model

```bash
python supervised_models/tabular_model/inference.py \
--test_csv data/tabular_features_w_text_test.csv \
--checkpoint_path supervised_models/tabular_model/training_logs/20250310_200642/best_model_iter_3450.pth
```
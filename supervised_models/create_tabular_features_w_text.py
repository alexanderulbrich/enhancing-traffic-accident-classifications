import argparse
from functools import reduce
import pandas as pd
import os

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Create tabular features for a multimodal model.")
    parser.add_argument(
        "--unfall_table_csv_path",
        type=str,
        required=True,
        help="Path to the unfall CSV file (e.g., data/D_Unfall_RData.csv)."
    )
    parser.add_argument(
        "--beteiligte_table_csv_path",
        type=str,
        required=True,
        help="Path to the beteiligte CSV file (e.g., data/D_Beteiligte_RData.csv)."
    )
    parser.add_argument(
        "--unfalltext_csv_path", 
        type=str, 
        required=True, 
        help="Path to the unfall_text CSV file (e.g., data/D_Unfalltext_image.csv)."
    )
    parser.add_argument(
        "--llm_labels_csv_path", 
        type=str,  
        help="Path to the llm few shot labels CSV file (e.g., data/few_shot_llm_predicted_labels_unfalltext_0_103111.csv)."
    )
    parser.add_argument(
        "--test_csv_path", 
        type=str, 
        help="Path to the test CSV file to remove from training (e.g., data/large_test_set.csv)."
    )
    parser.add_argument(
        "--duplicates_to_remove_excel_path",
        type=str,
        help="Path to the Excel file containing the duplicates to remove from the training set."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='data',
        help="Path to the output directory where the features CSV file will be saved."
    )    
    args = parser.parse_args()

    # ----------------------------
    # Read CSV Files
    # ----------------------------
    unfall_table = pd.read_csv(args.unfall_table_csv_path)
    beteiligte_table = pd.read_csv(args.beteiligte_table_csv_path)
    df_unfalltext = pd.read_csv(args.unfalltext_csv_path)

    # ----------------------------
    # Create Features from the Unfall Table
    # ----------------------------

    # Numerical features: use selected columns and rename (prefixing with "numerical_" except UN_KEY)
    unfall_features_numeric = unfall_table[["UN_KEY", "US_GEO_X", "US_GEO_Y", "KO_BETEIL"]].copy()
    unfall_features_numeric.columns = ['UN_KEY'] + [
        f'numerical_{col}' if col != 'UN_KEY' else col for col in unfall_features_numeric.columns[1:]
    ]
    print("Unfall numeric features shape:", unfall_features_numeric.shape)

    # Binary features: rename columns with prefix "categorical_"
    unfall_features_binary = unfall_table[["UN_KEY", "SS_FLUCHTG", "SS_DROGEN", "SS_ALKOH"]].copy()
    unfall_features_binary.columns = ["UN_KEY"] + [
        "categorical_" + col for col in unfall_features_binary.columns[1:]
    ]
    print("Unfall binary features shape:", unfall_features_binary.shape)

    # Month as a categorical feature: extract month from KO_UDATUM
    unfall_features_month = unfall_table[["UN_KEY", "KO_UDATUM"]].copy()
    unfall_features_month["KO_UDATUM"] = pd.to_datetime(unfall_features_month["KO_UDATUM"])
    unfall_features_month["categorical_KO_UDATUM_month"] = unfall_features_month["KO_UDATUM"].dt.month
    unfall_features_month = unfall_features_month.drop(columns=["KO_UDATUM"])
    print("Unfall month features shape:", unfall_features_month.shape)

    # KO_TYP categorical feature
    unfall_features_kotype = unfall_table[["UN_KEY", "KO_TYP"]].copy()
    unfall_features_kotype["categorical_KO_TYP"] = unfall_features_kotype["KO_TYP"]
    unfall_features_kotype = unfall_features_kotype.drop(columns=["KO_TYP"])
    print("Unfall KO_TYP features shape:", unfall_features_kotype.shape)

    # KO_USTDE_daytime categorical feature: split day into 4 parts
    unfall_features_koustde = unfall_table[["UN_KEY", "KO_USTDE"]].copy()
    unfall_features_koustde["categorical_KO_USTDE_daytime"] = pd.cut(
        unfall_features_koustde["KO_USTDE"], bins=4, labels=[1, 2, 3, 4]
    )
    unfall_features_koustde = unfall_features_koustde.drop(columns=["KO_USTDE"])
    print("Unfall KO_USTDE_daytime features shape:", unfall_features_koustde.shape)

    # ----------------------------
    # Create Features from the Beteiligte Table
    # ----------------------------

    # Average age of involved people
    beteiligte_features_avg_age = beteiligte_table.groupby("UN_KEY")["BT_ALTER"].mean().reset_index()
    beteiligte_features_avg_age.rename(columns={"BT_ALTER": "numerical_BT_ALTER_avg"}, inplace=True)
    print("Beteiligte average age features shape:", beteiligte_features_avg_age.shape)

    # Count of involved people
    beteiligte_features_count_involved_people = beteiligte_table.groupby("UN_KEY")["BT_NR"].count().reset_index()
    beteiligte_features_count_involved_people.rename(columns={"BT_NR": "numerical_BT_count"}, inplace=True)
    print("Beteiligte count features shape:", beteiligte_features_count_involved_people.shape)

    # Consequences counts
    temp_df = beteiligte_table.copy()
    temp_df['BT_FOLGEN_died'] = (temp_df['BT_FOLGEN'] == 1.0).astype(int)
    temp_df['BT_FOLGEN_severly_injured'] = (temp_df['BT_FOLGEN'] == 2.0).astype(int)
    temp_df['BT_FOLGEN_mildly_injured'] = (temp_df['BT_FOLGEN'] == 3.0).astype(int)
    beteiligte_features_consequences = (
        temp_df
        .groupby('UN_KEY', as_index=False)[['BT_FOLGEN_died', 'BT_FOLGEN_severly_injured', 'BT_FOLGEN_mildly_injured']]
        .sum()
    )
    beteiligte_features_consequences.rename(columns={
        'BT_FOLGEN_died': 'numerical_BT_FOLGEN_died',
        'BT_FOLGEN_severly_injured': 'numerical_BT_FOLGEN_severly_injured',
        'BT_FOLGEN_mildly_injured': 'numerical_BT_FOLGEN_mildly_injured'
    }, inplace=True)
    print("Beteiligte consequences features shape:", beteiligte_features_consequences.shape)

    # Involved vehicle type counts
    temp_df = beteiligte_table.copy()
    categories = list(temp_df.BT_BETEIL_K.unique())
    for cat in categories:
        temp_df[f'BT_BETEIL_K_{cat}'] = (temp_df['BT_BETEIL_K'] == cat).astype(int)
    beteiligte_features_vehicle_type_counts = temp_df.groupby(
        'UN_KEY', as_index=False
    )[[f'BT_BETEIL_K_{cat}' for cat in categories]].sum()
    beteiligte_features_vehicle_type_counts.rename(
        columns={f'BT_BETEIL_K_{cat}': f'numerical_BT_BETEIL_K_{cat}' for cat in categories},
        inplace=True
    )
    # Rename specific columns if necessary
    beteiligte_features_vehicle_type_counts.rename(
        columns={
            'numerical_BT_BETEIL_K_Sonstige und unbekannte Fahrzeuge (Unfallflucht)': 'numerical_BT_BETEIL_K_Sonstige',
            'numerical_BT_BETEIL_K_Andere Kfz': 'numerical_BT_BETEIL_K_Andere_Kfz'
        },
        inplace=True
    )
    print("Beteiligte vehicle type counts features shape:", beteiligte_features_vehicle_type_counts.shape)

    # Percentage of available Ausweis (ID)
    temp_df = beteiligte_table.copy()
    temp_df['BTA_AUSW'] = temp_df['BTA_AUSW'].replace(2, 0)
    beteiligte_features_avg_ausweis = temp_df.groupby("UN_KEY")["BTA_AUSW"].mean().reset_index()
    beteiligte_features_avg_ausweis.rename(columns={"BTA_AUSW": "numerical_BTA_AUSW_percentage"}, inplace=True)
    print("Beteiligte ausweis percentage features shape:", beteiligte_features_avg_ausweis.shape)

    # Participants sex counts (ignoring category 3 since there are only 2 examples for it)
    temp_df = beteiligte_table.copy()
    temp_df['BT_SEX_male'] = (temp_df['BT_SEX'] == 1).astype(int)
    temp_df['BT_SEX_female'] = (temp_df['BT_SEX'] == 2).astype(int)
    beteiligte_features_sex_counts = temp_df.groupby('UN_KEY', as_index=False)[['BT_SEX_male', 'BT_SEX_female']].sum()
    beteiligte_features_sex_counts.rename(
        columns={'BT_SEX_male': 'numerical_BT_SEX_male', 'BT_SEX_female': 'numerical_BT_SEX_female'},
        inplace=True
    )
    print("Beteiligte sex counts features shape:", beteiligte_features_sex_counts.shape)

    # ----------------------------
    # Merge All Features
    # ----------------------------
    pd.set_option('display.max_columns', None)

    feature_tables = [
        unfall_features_numeric,
        unfall_features_binary,
        unfall_features_month,
        unfall_features_kotype,
        unfall_features_koustde,
        beteiligte_features_avg_age,
        beteiligte_features_count_involved_people,
        beteiligte_features_consequences,
        beteiligte_features_vehicle_type_counts,
        beteiligte_features_avg_ausweis,
        beteiligte_features_sex_counts
    ]

    features = reduce(lambda left, right: pd.merge(left, right, on="UN_KEY", how="inner"), feature_tables)
    print("Merged features shape:", features.shape)
    features.info()

    # ----------------------------
    # Fill Missing Values
    # ----------------------------
    features["numerical_US_GEO_X"].fillna(0, inplace=True)
    features["numerical_US_GEO_Y"].fillna(0, inplace=True)
    features["numerical_BT_ALTER_avg"].fillna(features["numerical_BT_ALTER_avg"].mean(), inplace=True)
    if features.isna().sum().sum() == 0:
        print("No missing values remain in the features DataFrame.")
    else:
        print("There are still missing values in the features DataFrame.")

    # Count and print the number of categorical and numerical columns (by prefix)
    categorical_count = sum(col.startswith('categorical_') for col in features.columns)
    numerical_count = sum(col.startswith('numerical_') for col in features.columns)
    print(f"Categorical columns count: {categorical_count}")
    print(f"Numerical columns count: {numerical_count}")

    # ----------------------------
    # One-Hot Encoding for Categorical Features with More Than Two Categories
    # ----------------------------
    categorical_KO_UDATUM_dummies = pd.get_dummies(
        features['categorical_KO_UDATUM_month'], prefix='categorical_KO_UDATUM_month'
    ).astype(int)
    categorical_KO_TYP_dummies = pd.get_dummies(
        features['categorical_KO_TYP'], prefix='categorical_KO_TYP'
    ).astype(int)
    categorical_KO_USTDE_daytime_dummies = pd.get_dummies(
        features['categorical_KO_USTDE_daytime'], prefix='categorical_KO_USTDE_daytime'
    ).astype(int)

    # Concatenate the new dummy columns and remove the originals
    features = pd.concat(
        [features, categorical_KO_UDATUM_dummies, categorical_KO_TYP_dummies, categorical_KO_USTDE_daytime_dummies],
        axis=1
    )
    features.drop(columns=['categorical_KO_UDATUM_month', 'categorical_KO_TYP', 'categorical_KO_USTDE_daytime'], inplace=True)

    print("Tabular features shape after one-hot encoding:", features.shape)
    
    # ----------------------------
    # Add the accident text description column
    # ----------------------------
    features = pd.merge(features, df_unfalltext[["UN_KEY", "Text"]], on="UN_KEY", how="inner")
    print("Tabular features with Text shape:", features.shape)

    # Add the label Unfalltyp
    features = pd.merge(features, unfall_table[["UN_KEY", "Unfalltyp"]], on="UN_KEY", how="inner")
    print("===Tabular Features and Labels Preview===")
    print(features.head(3))

    # Save features to a CSV file for all data
    print("All Data Shape:", features.shape)
    output_path = os.path.join(args.output_dir, "tabular_features_w_text_all_data.csv")
    features.to_csv(output_path, index=False)

    if args.duplicates_to_remove_excel_path:
        # Remove duplicates from the training set
        duplicates_to_remove = pd.read_excel(args.duplicates_to_remove_excel_path)
        duplicates_to_remove = duplicates_to_remove[duplicates_to_remove["filter"]=="yes"]["Text"].tolist()
        features = features[~features["Text"].isin(duplicates_to_remove)].copy().reset_index(drop=True)

    if args.test_csv_path:
        # Remove the test set from the training set
        test_set = pd.read_csv(args.test_csv_path)
        # Save the test features to a CSV file
        test_features = features[features["UN_KEY"].isin(test_set["UN_KEY"])]
        print("Test Data Shape:", test_features.shape)
        test_output_path = os.path.join(args.output_dir, "tabular_features_w_text_test.csv")
        test_features.to_csv(test_output_path, index=False)

        # Remove the test set from the training set
        features = features[~features["UN_KEY"].isin(test_set["UN_KEY"])]
    
    if args.llm_labels_csv_path:
        # Get the intersection of human and llm labels
        llm_labels = pd.read_csv(args.llm_labels_csv_path)
        features = pd.merge(features, llm_labels[["UN_KEY", "few_shot_LLM_Labels"]], on="UN_KEY", how="inner")

        # Only Human Label 7 data
        features_w_human_labels_7 = features[features['Unfalltyp'] == 7].copy().reset_index(drop=True)
        features_w_human_labels_7.drop(columns=["few_shot_LLM_Labels"], inplace=True)
        print("Training Data with Human Labels 7 Shape:", features_w_human_labels_7.shape)
        output_path = os.path.join(args.output_dir, "tabular_features_w_text_inference_human_labels_7.csv")
        features_w_human_labels_7.to_csv(output_path, index=False)
        
        # Data with Low Quality Labels
        condition = (
            (features['Unfalltyp'].isin(range(1, 7))) |
            ((features['Unfalltyp'] == 7) & (features['few_shot_LLM_Labels'] == 7))
        )
        features_w_low_quality_labels = features[condition].copy().reset_index(drop=True)
        features_w_low_quality_labels.drop(columns=["few_shot_LLM_Labels"], inplace=True)
        print("Training Data with Low Quality Labels Shape:", features_w_low_quality_labels.shape)
        output_path = os.path.join(args.output_dir, "tabular_features_w_text_train_low_quality_labels.csv")
        features_w_low_quality_labels.to_csv(output_path, index=False)        

        # Data with High Quality Labels
        features_w_high_quality_labels = features[features["few_shot_LLM_Labels"]==features["Unfalltyp"]].copy().reset_index(drop=True)
        # Remove the few_shot_LLM_Labels column
        features_w_high_quality_labels.drop(columns=["few_shot_LLM_Labels"], inplace=True)

    # Save features to a CSV file for training data
    print("Training Data with High Quality Labels Shape:", features_w_high_quality_labels.shape)
    print("===Data Preview===")
    print(features_w_high_quality_labels.head(3))
    output_path = os.path.join(args.output_dir, "tabular_features_w_text_train.csv")
    features_w_high_quality_labels.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()

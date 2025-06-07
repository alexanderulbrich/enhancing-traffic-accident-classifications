# Load necessary libraries
library(tibble, lib.loc = "/home/ra32qov/R/x86_64-pc-linux-gnu-library/4.3")
library(readxl, lib.loc = "/home/ra32qov/R/x86_64-pc-linux-gnu-library/4.3")

# Load the dataset
load("/home/accidents/data/2017–2022.image")

# List the names of the specific objects you want to export
selected_objects <- c("D_Beteiligte", "D_Fahrzeug", "D_Mitfahrer", "D_UN_Codes", "D_Unfall", "D_Unfalltext")

# Loop through each object in the environment
for (obj_name in selected_objects) {
  obj <- get(obj_name)
  
  # Check if the object is a data frame
  if (is.data.frame(obj)) {
    # Export the data frame to a CSV file
    write.csv(obj, file = paste0("data/", obj_name, "_image.csv"), row.names = FALSE)
  }
}

# Read the duplicated text description list
duplicated_texts <- read_excel("/home/accidents/data/contributions_wise2425/duplicated_text_description_list.xlsx")

# Filter the texts with "yes" in the filter column
texts_to_remove <- duplicated_texts$Text[duplicated_texts$filter == "yes"]

# Randomly sample 50,000 rows from D_Unfalltext after removing filtered texts
if ("D_Unfalltext" %in% selected_objects) {
  set.seed(123)  # Set a seed for reproducibility
  D_Unfalltext <- get("D_Unfalltext")
  
  # Ensure the object is a data frame
  if (is.data.frame(D_Unfalltext)) {
    # Remove rows where the Text column matches any value in texts_to_remove
    D_Unfalltext_filtered <- D_Unfalltext[!D_Unfalltext$Text %in% texts_to_remove, ]
    
    # Sample 50,000 rows or all rows if fewer than 50,000
    D_Unfalltext_sample <- D_Unfalltext_filtered[sample(nrow(D_Unfalltext_filtered), min(50000, nrow(D_Unfalltext_filtered))), ]
    
    # Save the sampled data to a CSV file
    write.csv(D_Unfalltext_sample, file = "data/D_Unfalltext_image_50000_subsample.csv", row.names = FALSE)
  }
}

# Remove the .image objects
rm(list = ls())

# Load the dataset
load("/home/accidents/data/Image_Export_Full_2017bis2022.RData")

# List the names of the specific objects you want to export
selected_objects <- c("D_Beteiligte", "D_Fahrzeug", "D_Mitfahrer", "D_UN_Codes", "D_Unfall")

# Loop through each object in the environment
for (obj_name in selected_objects) {
  obj <- get(obj_name)
  
  # Check if the object is a data frame
  if (is.data.frame(obj)) {
    # Export the data frame to a CSV file
    write.csv(obj, file = paste0("data/", obj_name, "_RData.csv"), row.names = FALSE)
  }
}


# Load the old consulting project predictions
load("data/consulting_project_2023_classification_results.RData")

# Check if the dataframe exists and save it as a CSV
if (exists("new_classification_results")) {
  # Ensure the object is a data frame
  if (is.data.frame(new_classification_results)) {
    # Save the dataframe to a CSV file
    write.csv(new_classification_results, file = "data/old_consulting_project_2023_classification_results.csv", row.names = FALSE)
  } else {
    warning("new_classification_results is not a data frame.")
  }
} else {
  warning("The object new_classification_results does not exist in the environment.")
}
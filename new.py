import pandas as pd

def extract_rows_with_wk_gt_10(file_path):
    """
    Extracts rows from a Parquet file where the 'wk' column value is greater than 10.
    
    Args:
        file_path (str): Path to the Parquet file
        
    Returns:
        pd.DataFrame: DataFrame containing only rows where wk > 10
    """
    try:
        # Read the Parquet file
        df = pd.read_parquet(file_path)
        
        # Check if 'wk' column exists
        if 'wk' not in df.columns:
            raise ValueError("The 'wk' column does not exist in the dataset.")
            
        # Filter rows where wk > 10
        filtered_df = df[df['wk'] > 10]

        filtered_df.to_csv('filtered_results.csv')
        
        return filtered_df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = r"dataset_files\rect_section\datasetSGU.parquet"
    result = extract_rows_with_wk_gt_10(file_path)
    
    if result is not None:
        print(f"Found {len(result)} rows where wk > 10")
        print(result.head())  # Display first few rows of the result
    else:
        print("No results returned.")

    
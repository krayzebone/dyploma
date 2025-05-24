import pandas as pd

def convert_parquet_to_xlsx(input_parquet, output_xlsx, n_samples=10000):
    """
    Convert a Parquet file to Excel (.xlsx) with a limited number of samples.
    
    Args:
        input_parquet (str): Path to input Parquet file
        output_xlsx (str): Path for output Excel file
        n_samples (int): Number of samples to include (default: 10000)
    """
    # Read Parquet file (only loading the first n_samples rows)
    df = pd.read_parquet(input_parquet).head(n_samples)
    
    # Save to Excel
    df.to_excel(output_xlsx, index=False)
    print(f"Successfully saved first {len(df)} samples to {output_xlsx}")

# Example usage:
if __name__ == "__main__":
    input_file = r"neural_networks\rect_section_n2\dataset\dataset_testnoas2.parquet"  # Change to your input file
    output_file = "output_data.xlsx"   # Change to your desired output file
    
    convert_parquet_to_xlsx(input_file, output_file)
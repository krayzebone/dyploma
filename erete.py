import pandas as pd
import pyarrow.parquet as pq

def parquet_to_excel(parquet_path, excel_path, sheet_name='Sheet1'):
    """
    Convert a Parquet file to Excel format.
    
    Parameters:
        parquet_path (str): Path to the input Parquet file
        excel_path (str): Path where the Excel file should be saved
        sheet_name (str): Name of the Excel sheet (default: 'Sheet1')
    """
    try:
        # Read the Parquet file
        df = pd.read_parquet(parquet_path)
        
        # Write to Excel
        df.to_excel(excel_path, sheet_name=sheet_name, index=False)
        
        print(f"Successfully converted {parquet_path} to {excel_path}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

# Example usage
if __name__ == "__main__":
    input_file = r"dataset_files\rect_section\dataset_rect_section.parquet"  # Replace with your input file path
    output_file = r"dataset_files\rect_section\output.xlsx"   # Replace with your desired output path
    
    parquet_to_excel(input_file, output_file)
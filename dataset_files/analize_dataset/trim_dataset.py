import pandas as pd

def sample_parquet(input_path, output_path, n_samples=100000, random_state=42):
    """
    Randomly sample n_samples from a Parquet file.
    If file has <= n_samples, keeps all rows unchanged.
    
    Args:
        input_path: Path to input Parquet file
        output_path: Path to save sampled Parquet file
        n_samples: Number of samples to keep (default: 100000)
        random_state: Random seed for reproducibility (default: 42)
    """
    df = pd.read_parquet(input_path)
    
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=random_state)
        print(f"Downsampled from {len(df)} to {n_samples} rows")
    else:
        print(f"File already has {len(df)} rows (≤ {n_samples}), keeping unchanged")
    
    df.to_parquet(output_path)
    print(f"Saved sampled data to {output_path}")

# Example usage
sample_parquet(
    input_path=r"dataset_files\Tsectionplus\Tsection1.parquet",
    output_path="Tsection1_100k.parquet",
    n_samples=100000
)
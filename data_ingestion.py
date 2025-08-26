import pandas as pd
import os

def ingest_data(file_path):
    """
    Ingest Adult Income dataset, handle basics, save raw and cleaned versions.
    """
    # Column names (from UCI documentation)
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    
    # Load raw data (no header in adult.data)
    raw_df = pd.read_csv(file_path, names=columns, skipinitialspace=True)
    
    # Save raw data
    os.makedirs('artifacts', exist_ok=True)
    raw_df.to_csv(os.path.join('artifacts', 'raw_data.csv'), index=False)
    
    # Handle duplicates
    cleaned_df = raw_df.drop_duplicates()
    
    # Basic cleaning: Replace '?' with NA and drop rows with missing values
    cleaned_df = cleaned_df.replace('?', pd.NA)
    cleaned_df.dropna(inplace=True)
    
    # Convert income to binary (0: <=50K, 1: >50K)
    cleaned_df['income'] = cleaned_df['income'].map({'<=50K': 0, '>50K': 1})
    
    # Drop fnlwgt (often not predictive)
    cleaned_df = cleaned_df.drop('fnlwgt', axis=1)
    
    # Save cleaned data
    cleaned_df.to_csv(os.path.join('artifacts', 'cleaned_data.csv'), index=False)
    
    return cleaned_df

if __name__ == "__main__":
    ingest_data(os.path.join('data', 'adult.data'))
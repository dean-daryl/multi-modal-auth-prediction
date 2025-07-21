import pandas as pd
import os

# File paths
SOCIAL_PROFILES_PATH = "../data/Customer Social Profiles.csv"
TRANSACTIONS_PATH = "../data/Customer Transactions.csv"
OUTPUT_PATH = "../data/merged_data.csv"

def load_data():
    profiles = pd.read_csv(SOCIAL_PROFILES_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH)
    return profiles, transactions

def normalize_customer_ids(profiles, transactions):
    # Remove leading 'A' from customer_id_new, convert to string, strip whitespace, lower-case
    profiles['customer_id_new'] = profiles['customer_id_new'].astype(str).str.strip().str.upper().str.replace('A', '', n=1)
    transactions['customer_id_legacy'] = transactions['customer_id_legacy'].astype(str).str.strip().str.upper()
    return profiles, transactions

def merge_data(profiles, transactions):
    # Merge on normalized customer IDs
    merged = pd.merge(
        profiles,
        transactions,
        left_on='customer_id_new',
        right_on='customer_id_legacy',
        how='outer'
    )
    return merged

def handle_missing_data(df):
    # Fill numeric columns with median, categorical with 'Unknown'
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna('Unknown', inplace=True)
    return df

def normalize_formats(df):
    # Example: Standardize date columns if present
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def one_hot_encode(df):
    # Exclude ID columns from encoding
    id_cols = ['customer_id_new', 'customer_id_legacy']
    categorical_cols = df.select_dtypes(include=['object']).columns.difference(id_cols)
    df = pd.get_dummies(df, columns=categorical_cols)
    return df

def main():
    print("Loading data...")
    profiles, transactions = load_data()
    print("Normalizing customer IDs...")
    profiles, transactions = normalize_customer_ids(profiles, transactions)
    print("Merging datasets...")
    merged = merge_data(profiles, transactions)
    print(f"Merged dataset shape: {merged.shape}")
    print("Handling missing data...")
    merged = handle_missing_data(merged)
    print("Normalizing formats...")
    merged = normalize_formats(merged)
    print("One-hot encoding categorical fields...")
    merged = one_hot_encode(merged)
    print(f"Final dataset shape: {merged.shape}")
    print(f"Saving merged dataset to {OUTPUT_PATH} ...")
    merged.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()

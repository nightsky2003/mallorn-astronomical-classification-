import pandas as pd
import numpy as np
import os
import sys
import subprocess

# --- AUTO-INSTALL EXTINCTION LIBRARY ---
try:
    import extinction
    print("✅ 'extinction' library found.")
except ImportError:
    print("⚠️ 'extinction' library not found. Installing it for you...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "extinction==0.4.7"])
    import extinction
    print("✅ Installation complete.")

# =================CONFIGURATION=================
NUM_SPLITS = 20
SPLIT_FOLDER_PREFIX = "split_"
TRAIN_LOG_FILE = "train_log.csv"
TEST_LOG_FILE = "test_log.csv"

WAVELENGTHS = {
    'u': 3641, 'g': 4704, 'r': 6155, 
    'i': 7504, 'z': 8695, 'y': 10056
}
# ===============================================

def calculate_extinction_factors():
    print("\n--- Pre-calculating Physics Constants ---")
    factors = {}
    for band, wave in WAVELENGTHS.items():
        factor = extinction.fitzpatrick99(np.array([wave]), 1.0)[0]
        factors[band] = factor
    return factors

def process_data(log_filename, type_prefix, ext_factors):
    print(f"\n🚀 STARTING {type_prefix.upper()} DATA PROCESSING")
    
    if not os.path.exists(log_filename):
        print(f"❌ Error: {log_filename} not found.")
        return None

    log_df = pd.read_csv(log_filename)
    
    if 'A_v' not in log_df.columns and 'EBV' in log_df.columns:
        log_df['A_v'] = log_df['EBV'] * 3.1
    
    keep_cols = ['object_id', 'A_v', 'Z', 'Z_err', 'EBV', 'target', 'truth', 'SpecType']
    keep_cols = [c for c in keep_cols if c in log_df.columns]
    
    all_chunks = []

    for i in range(1, NUM_SPLITS + 1):
        folder_name = f"{SPLIT_FOLDER_PREFIX}{i:02d}"
        file_name = f"{type_prefix}_full_lightcurves.csv"
        file_path = os.path.join(folder_name, file_name)
        
        # Fallback for flat directory structure
        if not os.path.exists(file_path):
             if os.path.exists(file_name):
                 file_path = file_name
        
        if os.path.exists(file_path):
            print(f"  📂 Processing {file_path}...", end="\r")
            
            chunk = pd.read_csv(file_path)
            chunk = pd.merge(chunk, log_df[keep_cols], on="object_id", how="inner")
            
            # --- PHYSICS CORRECTION ---
            chunk['ext_factor'] = chunk['Filter'].map(ext_factors)
            chunk['A_lambda'] = chunk['ext_factor'] * chunk['A_v']
            
            # Correction Factor (Linear scale)
            correction = 10 ** (chunk['A_lambda'] / 2.5)
            
            # Apply to Flux AND Flux_err (Crucial for SNR calculations)
            chunk['Flux_Corrected'] = chunk['Flux'] * correction
            chunk['Flux_err_Corrected'] = chunk['Flux_err'] * correction
            
            chunk = chunk.drop(columns=['ext_factor', 'A_lambda'])
            all_chunks.append(chunk)

    if all_chunks:
        print(f"\n  🧩 Merging {len(all_chunks)} chunks...")
        final_df = pd.concat(all_chunks, ignore_index=True)
        return final_df
    
    print(f"\n❌ No data found for {type_prefix}.")
    return None

if __name__ == "__main__":
    ext_factors = calculate_extinction_factors()
    
    train_df = process_data(TRAIN_LOG_FILE, "train", ext_factors)
    if train_df is not None:
        train_df.to_parquet("master_train_corrected.parquet", index=False)
        print("💾 Saved to 'master_train_corrected.parquet'")

    test_df = process_data(TEST_LOG_FILE, "test", ext_factors)
    if test_df is not None:
        test_df.to_parquet("master_test_corrected.parquet", index=False)
        print("💾 Saved to 'master_test_corrected.parquet'")
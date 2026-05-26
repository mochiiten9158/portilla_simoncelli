import os
import numpy as np
import subprocess
from pathlib import Path
import pandas as pd
import shutil

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.feature_selection import f_classif

import warnings
warnings.filterwarnings("ignore")

from generate_plots_rensink import generate_harrison_scatter
from generate_pc_rensink import generate_harrison_pcp

method = "positive parallel coordinates"

def generate_synthetic_data_training(method, rbase = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
    if method == "positive scatterplots":
        corrs_positive = [round(i * 0.0025, 4) for i in range(0, 401)]
        count = 1
        for r in rbase:
            count = 1
            generate_harrison_scatter(corr=r, seed=None,
                                      out_dir=f"assets/training/{r}", count=count)
            count += 1
            for i in range(5):
                for corr in corrs_positive:
                    if (corr != r):
                        generate_harrison_scatter(corr=corr, seed=None,
                                              out_dir=f"assets/training/{r}", count=count)
                        count += 1

    elif method == "positive parallel coordinates":
        corrs_positive = [round(i * 0.0025, 4) for i in range(0, 401)]
        count = 1
        for r in rbase:
            count = 1
            generate_harrison_pcp(corr=r, seed=None,
                                      out_dir=f"assets/training/{r}", count=count)
            count += 1
            for i in range(5):
                for corr in corrs_positive:
                    if (corr != r):
                        generate_harrison_pcp(corr=corr, seed=None,
                                              out_dir=f"assets/training/{r}", count=count)
                        count += 1

def generate_synthetic_data_testing(method, rbase = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
    if method == "positive scatterplots":
        corrs_positive = [round(i * 0.0025, 4) for i in range(0, 401)]
        count = 1
        print(f"count: {count}") # get rid of this print statement, just for testing
        for r in rbase:
            count = 1
            generate_harrison_scatter(corr=r, seed=None,
                                      out_dir=f"assets/testing/{r}", count=count)
            count += 1
            for corr in corrs_positive:
                if (corr != r):
                    generate_harrison_scatter(corr=corr, seed=None,
                                      out_dir=f"assets/testing/{r}", count=count)
                    count += 1
    
    elif method == "positive parallel coordinates":
        corrs_positive = [round(i * 0.0025, 4) for i in range(0, 401)]
        count = 1
        for r in rbase:
            count = 1
            generate_harrison_pcp(corr=r, seed=None,
                                      out_dir=f"assets/testing/{r}", count=count)
            count += 1
            for corr in corrs_positive:
                if (corr != r):
                    generate_harrison_pcp(corr=corr, seed=None,
                                      out_dir=f"assets/testing/{r}", count=count)
                    count += 1


def extract_ps_stats_from_images(input_dir="assets/training", stats_bin="./extract_stats", output_base_dir="summary_stats_training"):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
        
    if not os.path.exists(stats_bin):
        print(f"Error: C++ binary '{stats_bin}' not found.")
        return

    # corr_regex = re.compile(r"(-?\d+\.\d+)")
    rbase_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    if not rbase_dirs:
        print(f"No subfolders found inside {input_dir}")
        return
    print(f"Found {len(rbase_dirs)} correlation base directories in {input_dir}.")

    for rbase_dir in rbase_dirs:
        images = sorted(list(rbase_dir.glob("*.png")))
        total_images = len(images)
        if total_images == 0:
            print(f"No PNG images found in {rbase_dir}. Skipping this directory.")
            continue

        print(f"Processing {len(images)} images in {rbase_dir}...")

        current_out_dir = Path(output_base_dir) / rbase_dir.name
        current_out_dir.mkdir(parents=True, exist_ok=True)

        for count, img_path in enumerate(images, start=1):
            img_name_stem = img_path.stem
            out_stats_path = current_out_dir / f"{img_name_stem}.csv"
            print(f"\r    [{count:3d}/{total_images:3d}] Processing {img_path.name}", end="", flush=True)
            try:
                subprocess.run([stats_bin, str(img_path), str(out_stats_path)],stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except subprocess.CalledProcessError:
                print(f"\n[ERRPR] C++ binary failed on image: {img_path.name}")

        print()

    print(f"\nDone. All statistics saved to base directory: '{output_base_dir}'")



def cleanup_training_images(training_dir="assets/training", force=False):
    training_path = Path(training_dir)
    
    if not training_path.exists():
        print(f"Cleanup skipped: '{training_dir}' does not exist.")
        return

    # If force is True, we skip the input prompt entirely
    if not force:
        # Double check we are not accidentally deleting important root directories
        confirm = input(f"Are you sure you want to delete all files in '{training_path.resolve()}'? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cleanup cancelled by user.")
            return

    print(f"Cleaning up files in {training_dir}...")
    try:
        # shutil.rmtree cleanly removes the folder and all contents inside it
        shutil.rmtree(training_path)
        print("Files removed successfully. Disk space cleared!")
    except Exception as e:
        print(f"An error occurred during cleanup: {e}")


def load_ps_stats(filepath):
    return np.loadtxt(filepath, delimiter=",", dtype=np.float32)


def extract_correlation_from_filename(filename):
    try:
        right_side = filename.split("_")[1]
        return float(right_side.replace('.csv', ''))
    except (IndexError, ValueError):
        raise ValueError(f"Could not extract correlation from filename: {filename}")


def process_folder(folder_path = "summary_stats_training"):
    base_path = Path(folder_path)
    if not base_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return None, None, None, None, None
    
    subfolders = sorted([d for d in base_path.iterdir() if d.is_dir()])

    global_X = []
    global_y = []
    global_corr_pairs = []

    rng = np.random.default_rng()

    for folder in subfolders:
        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {folder}. Skipping this directory.")
            continue

        print(f"Processing {len(csv_files)} CSV files in {folder.name}...")

        base_file = None
        comparison_files = []
        for file_path in csv_files:
            if file_path.name.startswith(f"1_{folder.name}"):
                base_file = file_path
            else:
                comparison_files.append(file_path)

        if base_file is None:
            print(f"[WARNING] Base file starting with '1_{folder.name}' not found in {folder.name}. Skipping folder.")
            continue
        
        base_stats = load_ps_stats(base_file)
        base_corr = extract_correlation_from_filename(base_file.name)

        for comp_file in comparison_files:
            comp_stats = load_ps_stats(comp_file)
            comp_corr = extract_correlation_from_filename(comp_file.name)

            left_stats, right_stats = base_stats, comp_stats
            left_corr, right_corr = base_corr, comp_corr
            label = 1 if abs(left_corr) > abs(right_corr) else 0

            if rng.random() < 0.5:
                left_stats, right_stats = right_stats, left_stats
                left_corr, right_corr = right_corr, left_corr
                label = 1 - label
            
            diff = left_stats - right_stats

            global_X.append(diff)
            global_y.append(label)
            global_corr_pairs.append((left_corr, right_corr))
    
    X_raw = np.array(global_X, dtype=np.float32)
    y_array = np.array(global_y, dtype=np.int32)
    corr_pairs_array = np.array(global_corr_pairs, dtype=np.float32)

    global_mu = np.mean(X_raw, axis=0)
    global_sigma = np.std(X_raw, axis=0)

    X_normalized = (X_raw - global_mu) / (global_sigma + 1e-9)

    return X_normalized, y_array, corr_pairs_array, global_mu, global_sigma


def process_folder_testing(folder_path = "summary_stats_testing", mu=None, sigma=None):
    base_path = Path(folder_path)
    if not base_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return None, None, None
    
    subfolders = sorted([d for d in base_path.iterdir() if d.is_dir()])

    global_X = []
    global_y = []
    global_corr_pairs = []

    rng = np.random.default_rng()

    for folder in subfolders:
        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {folder}. Skipping this directory.")
            continue

        print(f"Processing {len(csv_files)} CSV files in {folder.name}...")

        base_file = None
        comparison_files = []
        for file_path in csv_files:
            if file_path.name.startswith(f"1_{folder.name}"):
                base_file = file_path
            else:
                comparison_files.append(file_path)

        if base_file is None:
            print(f"[WARNING] Base file starting with '1_{folder.name}' not found in {folder.name}. Skipping folder.")
            continue
        
        base_stats = load_ps_stats(base_file)
        base_corr = extract_correlation_from_filename(base_file.name)

        for comp_file in comparison_files:
            comp_stats = load_ps_stats(comp_file)
            comp_corr = extract_correlation_from_filename(comp_file.name)

            left_stats, right_stats = base_stats, comp_stats
            left_corr, right_corr = base_corr, comp_corr
            label = 1 if abs(left_corr) > abs(right_corr) else 0

            if rng.random() < 0.5:
                left_stats, right_stats = right_stats, left_stats
                left_corr, right_corr = right_corr, left_corr
                label = 1 - label
            
            diff = left_stats - right_stats

            global_X.append(diff)
            global_y.append(label)
            global_corr_pairs.append((left_corr, right_corr))
    
    X_raw = np.array(global_X, dtype=np.float32)
    y_array = np.array(global_y, dtype=np.int32)
    corr_pairs_array = np.array(global_corr_pairs, dtype=np.float32)

    X_normalized = (X_raw - mu) / (sigma + 1e-9)

    return X_normalized, y_array, corr_pairs_array

print("\n--- Starting Training Pipeline --- on method: ", method)
print("Generating synthetic data...")
generate_synthetic_data_training(method=method)
print("\nExtracting summary statistics for TRAINING from generated images...")
extract_ps_stats_from_images(input_dir="assets/training", stats_bin="./extract_stats", output_base_dir="summary_stats_training")
print("\nStarting storage cleanup for TRAINING images...")
cleanup_training_images(training_dir="assets/training")

print("Processing summary statistics and preparing dataset...")
X, y, pairs, mean, sigma = process_folder(folder_path="summary_stats_training")
if X is not None:
        print("\n--- Pipeline Completed Successfully ---")
        print(f"X shape (Expected 4000 x 927 for 2 folders): {X.shape}")
        print(f"y shape (Expected 4000,):                   {y.shape}")
        print(f"Pairs tracking tracking tracker shape:      {pairs.shape}")
        print(f"Mean shape:                                 {mean.shape}")
        print(f"Sigma shape:                                {sigma.shape}")


X, y = shuffle(X, y, random_state=0)

model = Pipeline([
    ("scale", StandardScaler()),
    ("lda",   LinearDiscriminantAnalysis())
])

model.fit(X, y)

print("Finished training.")

num_participants = 100

all_corr1, all_corr2, all_correct, all_model_resp, all_participants = [], [], [], [], []

print("Starting testing simulation...")

for participant in range(1, num_participants + 1):
    print(f"\nSimulating participant {participant}/{num_participants}...")
    generate_synthetic_data_testing(method=method, rbase = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    extract_ps_stats_from_images(input_dir="assets/testing", stats_bin="./extract_stats", output_base_dir="summary_stats_testing")
    X_test, y_test, corr_test_pairs = process_folder_testing(folder_path="summary_stats_testing", mu=mean, sigma=sigma)
    if X_test is None:
        print(f"Error processing test folder for participant {participant}. Skipping this participant.")
        continue
    print(f"Test data for participant {participant} processed successfully. Starting predictions...")

    preds = model.predict(X_test)

    all_corr1.extend(corr_test_pairs[:, 0])
    all_corr2.extend(corr_test_pairs[:, 1])
    all_correct.extend(y_test)
    all_model_resp.extend(preds)
    all_participants.extend([participant] * len(y_test))

    cleanup_training_images(training_dir="assets/testing", force=True)
    cleanup_training_images(training_dir="summary_stats_testing", force=True)


all_corr1 = np.array(all_corr1)
all_corr2 = np.array(all_corr2)
all_correct = np.array(all_correct)
all_model_resp = np.array(all_model_resp)

print("Test Accuracy:", accuracy_score(all_correct, all_model_resp))

# Build DataFrame
df_sim = pd.DataFrame({
    "participant": all_participants,  # Iteration number acting as the user ID
    "corr1 (Left)": all_corr1,
    "corr2 (Right)": all_corr2,
    "correct": np.where(np.abs(all_corr1) > np.abs(all_corr2), "L", "R"),
    "modelResponse": np.where(all_model_resp == 1, "L", "R")
})

# Calculate performance diagnostics
df_sim["accuracy"] = (df_sim["correct"] == df_sim["modelResponse"]).astype(int)
df_sim["delta_corr"] = np.abs(df_sim["corr1 (Left)"] - df_sim["corr2 (Right)"])

# Quick validation check to guarantee the math adds up
print(f"Simulation complete. Total rows generated: {len(df_sim)}")

# Save Results
# df_sim.to_csv("test_pair_results.csv", index=False, float_format="%.4f")
# print("Saved test_pair_results.csv")


# df_sim.to_csv("test_pair_results_negative.csv", index=False, float_format="%.4f")
# print("Saved test_pair_results_negative.csv")


df_sim.to_csv("test_pair_results_pcp.csv", index=False, float_format="%.4f")
print("Saved test_pair_results_pcp.csv")



# df_sim.to_csv("test_pair_results_pcp_negative.csv", index=False, float_format="%.4f")
# print("Saved test_pair_results_pcp_negative.csv")
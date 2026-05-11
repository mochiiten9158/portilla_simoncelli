import os
import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")

# TRAINING_FOLDER       = "summary_stats_harrison_positive_100"
# TESTING_FOLDER        = "summary_stats_harrison_positive_100_0.0025"
# HARRISON_CSV = "harrison_results/scatter_positive.csv"



# TRAINING_FOLDER       = "summary_stats_harrison_negative_100"
# TESTING_FOLDER        = "summary_stats_harrison_negative_100_0.0025"
# HARRISON_CSV = "harrison_results/scatter_negative.csv"



# TRAINING_FOLDER       = "summary_stats_harrison_positive_pcp_100"
# TESTING_FOLDER        = "summary_stats_harrison_positive_pcp_100_0.0025"
# HARRISON_CSV = "harrison_results/parallelCoordinates_positive.csv"



TRAINING_FOLDER       = "summary_stats_harrison_negative_pcp_100"
TESTING_FOLDER        = "summary_stats_harrison_negative_pcp_100_0.0025"
HARRISON_CSV = "harrison_results/parallelCoordinates_negative.csv"

def load_ps_stats(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    numbers = re.findall(r'-?\d+\.\d+', text)
    return np.array(numbers, dtype=float)


def make_pairs(stats, correlations, bins):
    X, y, pair_types, corr_pairs = [], [], [], []
    N = len(stats)
    for i in range(N):
        for j in range(i + 1, N):
            diff  = stats[i] - stats[j]
            label = 1 if abs(correlations[i]) > abs(correlations[j]) else 0
            b1, b2 = bins[i], bins[j]
            X.append(diff);  y.append(label)
            pair_types.append((b1, b2))
            corr_pairs.append((correlations[i], correlations[j]))
            X.append(-diff); y.append(1 - label)
            pair_types.append((b1, b2))
            corr_pairs.append((correlations[j], correlations[i]))
    return (np.array(X), np.array(y),
            np.array(pair_types), np.array(corr_pairs))


training_stats_list   = []
training_correlations = []
testing_stats_list   = []
testing_correlations = []

for file in os.listdir(TRAINING_FOLDER):
    path  = os.path.join(TRAINING_FOLDER, file)
    parts = file.replace(".txt", "").split("_")
    corr  = float(parts[-2])
    training_stats_list.append(load_ps_stats(path))
    training_correlations.append(corr)

for file in os.listdir(TESTING_FOLDER):
    path  = os.path.join(TESTING_FOLDER, file)
    parts = file.replace(".txt", "").split("_")
    corr  = float(parts[-2])
    testing_stats_list.append(load_ps_stats(path))
    testing_correlations.append(corr)

train_stats        = np.array(training_stats_list)
test_stats         = np.array(testing_stats_list)
train_correlations = np.array(training_correlations)
test_correlations  = np.array(testing_correlations)
print("Training stats shape:", train_stats.shape)
print("Testing stats shape:", test_stats.shape)

num_bins  = 5
bin_edges = np.percentile(training_correlations, [0, 20, 40, 60, 80, 100])

bins_train = np.digitize(train_correlations, bin_edges[1:-1])
bins_test  = np.digitize(test_correlations,  bin_edges[1:-1])
corr_bins_train = np.digitize(train_correlations, bin_edges[1:-1])
corr_bins_test = np.digitize(test_correlations, bin_edges[1:-1])

X_train, y_train, type_train, _ = make_pairs(
    train_stats, train_correlations, bins_train
)

X_test, y_test, type_test, corr_test_pairs = make_pairs(
    test_stats, test_correlations, bins_test
)

X_test, y_test, type_test, corr_test_pairs = shuffle(
    X_test, y_test, type_test, corr_test_pairs,
    random_state=0
)

model = Pipeline([
    ("scale", StandardScaler()),
    ("pca",   PCA(n_components=0.95, whiten=False)),
    ("lda",   LinearDiscriminantAnalysis())
])


model.fit(X_train, y_train)
preds = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, preds))

# Extract correlations
corr1 = corr_test_pairs[:, 0]
corr2 = corr_test_pairs[:, 1]

correct = np.where(np.abs(corr1) > np.abs(corr2), "L", "R")

model_response = np.where(preds == 1, "L", "R")

accuracy = (model_response == correct).astype(int)

delta_corr = np.abs(corr2 - corr1)

# Build DataFrame
df = pd.DataFrame({
    "corr1 (Left)": corr1,
    "corr2 (Right)": corr2,
    "correct": correct,
    "modelResponse": model_response,
    "accuracy": accuracy,
    "delta_corr": delta_corr
})

# Save CSV
# df.to_csv("test_pair_results.csv", index=False, float_format="%.4f")

# print("Saved test_pair_results.csv")



# df.to_csv("test_pair_results_negative.csv", index=False, float_format="%.4f")

# print("Saved test_pair_results_negative.csv")



# df.to_csv("test_pair_results_pcp.csv", index=False, float_format="%.4f")

# print("Saved test_pair_results_pcp.csv")



df.to_csv("test_pair_results_pcp_negative.csv", index=False, float_format="%.4f")

print("Saved test_pair_results_pcp_negative.csv")
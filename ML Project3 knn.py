
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = r"C:\Users\madhu\OneDrive\Documents\ML PROJECT DATASETS\electronics.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

required_columns = [
    "item_id", "user_id", "rating",
    "category", "brand", "model_attr"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError("Missing columns: " + str(missing_columns))

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["user_id", "item_id", "rating"])

df["user_id"] = df["user_id"].astype(str)
df["item_id"] = df["item_id"].astype(str)
df = df[df["rating"].between(1, 5)]

print("\n========== DATASET DETAILS ==========")
print("Dataset Shape:", df.shape)
print("Unique Users:", df["user_id"].nunique())
print("Unique Products:", df["item_id"].nunique())
print("Rating Range:", df["rating"].min(), "-", df["rating"].max())
print("\nFirst Five Records:")
print(df.head())

# ==========================================
# 2. SELECT USERS WITH REPEATED INTERACTIONS
# ==========================================

user_counts = df["user_id"].value_counts()
repeated_users = user_counts[user_counts >= 2].index

df = df[df["user_id"].isin(repeated_users)].copy()

if len(df) < 100:
    raise ValueError(
        "Too few repeated-user interactions for reliable comparison."
    )

# Keep computation manageable for Spyder
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)

# Remove duplicate user-product pairs
df = (
    df.groupby(["user_id", "item_id"], as_index=False)
      .agg({
          "rating": "mean",
          "category": "first",
          "brand": "first",
          "model_attr": "first"
      })
)

df["liked"] = (df["rating"] >= 4).astype(int)

print("\n========== PREPARED DATA ==========")
print("Interactions:", len(df))
print("Users:", df["user_id"].nunique())
print("Products:", df["item_id"].nunique())

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

train, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

# Keep test records whose users and products appear in training
train_users = set(train["user_id"])
train_items = set(train["item_id"])

test = test[
    test["user_id"].isin(train_users) &
    test["item_id"].isin(train_items)
].copy()

if len(test) == 0:
    raise ValueError(
        "No known user-product pairs in test data. "
        "Try a different split or use more data."
    )

# Map user and product IDs to matrix indices
all_users = sorted(train["user_id"].unique())
all_items = sorted(train["item_id"].unique())

user_to_index = {u: i for i, u in enumerate(all_users)}
item_to_index = {p: i for i, p in enumerate(all_items)}

index_to_item = {i: p for p, i in item_to_index.items()}

train["u_idx"] = train["user_id"].map(user_to_index)
train["i_idx"] = train["item_id"].map(item_to_index)

test["u_idx"] = test["user_id"].map(user_to_index)
test["i_idx"] = test["item_id"].map(item_to_index)

n_users = len(all_users)
n_items = len(all_items)

R = csr_matrix(
    (
        train["rating"].to_numpy(),
        (train["u_idx"].to_numpy(), train["i_idx"].to_numpy())
    ),
    shape=(n_users, n_items),
    dtype=float
)

global_mean = train["rating"].mean()

user_means = train.groupby("u_idx")["rating"].mean().to_dict()
item_means = train.groupby("i_idx")["rating"].mean().to_dict()

print("\nTraining interactions:", len(train))
print("Testing interactions:", len(test))
print("User-Product Matrix:", R.shape)

# ==========================================
# 4. MODEL 1: SVD
# ==========================================

print("\n========== TRAINING SVD ==========")

n_components = min(20, n_users - 1, n_items - 1)

if n_components < 1:
    raise ValueError("Not enough users or products to train SVD.")

svd = TruncatedSVD(
    n_components=n_components,
    random_state=42,
    n_iter=7
)

user_factors = svd.fit_transform(R)
item_factors = svd.components_.T

def predict_svd(user_index, item_index):
    prediction = np.dot(
        user_factors[user_index],
        item_factors[item_index]
    )

    return float(np.clip(prediction, 1, 5))

print("SVD training completed.")
print("SVD Components:", n_components)
print("SVD Explained Variance:",
      round(svd.explained_variance_ratio_.sum(), 4))

# ==========================================
# 5. MODEL 2: ITEM-BASED KNN
# ==========================================

print("\n========== TRAINING KNN ==========")

# Each row represents one product and its user ratings
item_user_matrix = R.T.tocsr()

number_of_neighbors = min(21, n_items)

knn = NearestNeighbors(
    n_neighbors=number_of_neighbors,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

knn.fit(item_user_matrix)

neighbor_distances, neighbor_indices = knn.kneighbors(
    item_user_matrix
)

def predict_knn(user_index, item_index):
    neighbors = neighbor_indices[item_index]
    distances = neighbor_distances[item_index]

    user_ratings = R.getrow(user_index)
    rated_items = user_ratings.indices
    rated_values = user_ratings.data

    rating_lookup = dict(zip(rated_items, rated_values))

    weighted_sum = 0.0
    similarity_sum = 0.0

    for neighbor, distance in zip(neighbors, distances):
        if neighbor == item_index:
            continue

        if neighbor in rating_lookup:
            similarity = max(0.0, 1.0 - float(distance))

            weighted_sum += (
                similarity * rating_lookup[neighbor]
            )
            similarity_sum += similarity

    if similarity_sum > 0:
        prediction = weighted_sum / similarity_sum
    else:
        prediction = (
            user_means.get(user_index,
                           item_means.get(item_index, global_mean))
        )

    return float(np.clip(prediction, 1, 5))

print("KNN training completed.")
print("Neighbors used:", number_of_neighbors - 1)

# ==========================================
# 6. EVALUATE BOTH MODELS
# ==========================================

print("\n========== MODEL EVALUATION ==========")

y_true = test["liked"].to_numpy()

svd_predictions = []
knn_predictions = []

for row in test.itertuples(index=False):
    svd_predictions.append(
        predict_svd(row.u_idx, row.i_idx)
    )

    knn_predictions.append(
        predict_knn(row.u_idx, row.i_idx)
    )

svd_predictions = np.array(svd_predictions)
knn_predictions = np.array(knn_predictions)

# Rating >= 4 means recommended/liked
svd_labels = (svd_predictions >= 4).astype(int)
knn_labels = (knn_predictions >= 4).astype(int)

results = []

for model_name, predicted_labels in [
    ("SVD", svd_labels),
    ("KNN", knn_labels)
]:
    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, predicted_labels),
        "Precision": precision_score(
            y_true, predicted_labels, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predicted_labels, zero_division=0
        ),
        "F1-Score": f1_score(
            y_true, predicted_labels, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON TABLE")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("knn_svd_model_comparison.csv", index=False)

# ==========================================
# 7. SELECT THE BEST MODEL
# ==========================================

# F1 is the primary selection metric.
# Accuracy is used to break F1 ties.
best_row = results_df.sort_values(
    by=["F1-Score", "Accuracy"],
    ascending=False
).iloc[0]

best_model_name = best_row["Model"]

print("\n========== FINAL MODEL SELECTION ==========")
print("Best Model:", best_model_name)
print("Accuracy :", round(best_row["Accuracy"], 4))
print("Precision:", round(best_row["Precision"], 4))
print("Recall   :", round(best_row["Recall"], 4))
print("F1-Score :", round(best_row["F1-Score"], 4))

# ==========================================
# 8. PLOT METRIC COMPARISON
# ==========================================

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

results_plot = results_df.set_index("Model")[metrics]

ax = results_plot.T.plot(
    kind="bar",
    figsize=(10, 6),
    rot=0
)

ax.set_title("KNN vs SVD - Evaluation Metrics")
ax.set_xlabel("Evaluation Metrics")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend(title="Model")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("knn_svd_metric_comparison.png", dpi=300)
plt.show()

# ==========================================
# 9. PLOT F1-SCORE COMPARISON
# ==========================================

plt.figure(figsize=(7, 5))

plt.bar(
    results_df["Model"],
    results_df["F1-Score"]
)

plt.title("KNN vs SVD - F1-Score Comparison")
plt.xlabel("Model")
plt.ylabel("F1-Score")
plt.ylim(0, 1)

for i, score in enumerate(results_df["F1-Score"]):
    plt.text(i, score + 0.015, f"{score:.3f}", ha="center")

plt.tight_layout()
plt.savefig("knn_svd_f1_comparison.png", dpi=300)
plt.show()

# ==========================================
# 10. PLOT CONFUSION MATRICES
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ConfusionMatrixDisplay(
    confusion_matrix(y_true, knn_labels),
    display_labels=["Not Liked", "Liked"]
).plot(ax=axes[0], colorbar=False)

axes[0].set_title("KNN Confusion Matrix")

ConfusionMatrixDisplay(
    confusion_matrix(y_true, svd_labels),
    display_labels=["Not Liked", "Liked"]
).plot(ax=axes[1], colorbar=False)

axes[1].set_title("SVD Confusion Matrix")

plt.tight_layout()
plt.savefig("knn_svd_confusion_matrices.png", dpi=300)
plt.show()

# ==========================================
# 11. PREPARE PRODUCT AND BRAND INFORMATION
# ==========================================

product_info = (
    df.groupby("item_id")
      .agg(
          Average_Rating=("rating", "mean"),
          Rating_Count=("rating", "count"),
          Category=("category", "first"),
          Brand=("brand", "first"),
          Model_Attribute=("model_attr", "first")
      )
      .reset_index()
)

product_info["Brand"] = (
    product_info["Brand"].fillna("Unknown Brand").astype(str)
)

product_info["Category"] = (
    product_info["Category"].fillna("").astype(str)
)

product_info["Model_Attribute"] = (
    product_info["Model_Attribute"].fillna("").astype(str)
)

# Use products known to the trained models for personalized ranking
product_info["item_index"] = product_info["item_id"].map(item_to_index)

# ==========================================
# 12. PREDICT AND RECOMMEND PRODUCTS
# ==========================================

print("\n========== PRODUCT RECOMMENDATION ==========")

keyword = input(
    "Enter a product/category keyword (e.g., WATCH, HEADPHONES): "
).strip()

if not keyword:
    keyword = "electronics"

# Match the keyword against available product attributes
search_text = (
    product_info["Category"] + " " +
    product_info["Brand"] + " " +
    product_info["Model_Attribute"]
)

matched_products = product_info[
    search_text.str.contains(keyword, case=False, na=False, regex=False)
].copy()

# If the keyword does not match, show available categories and fallback
if matched_products.empty:
    print("\nNo exact keyword match was found.")
    print("Available categories:")
    print(
        product_info["Category"]
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(15)
        .to_string()
    )

    print("\nShowing top-rated products from the dataset instead.")
    matched_products = product_info.copy()

# Optional personalized recommendation
user_input = input(
    "\nEnter a known user_id for personalized recommendations "
    "(press Enter to skip): "
).strip()

if user_input and user_input in user_to_index:
    user_index = user_to_index[user_input]

    # Do not recommend products already rated by this user
    already_rated = set(R.getrow(user_index).indices)

    matched_products = matched_products[
        matched_products["item_index"].notna()
    ].copy()

    matched_products["item_index"] = (
        matched_products["item_index"].astype(int)
    )

    matched_products = matched_products[
        ~matched_products["item_index"].isin(already_rated)
    ].copy()

    if best_model_name == "SVD":
        matched_products["Predicted_Rating"] = matched_products[
            "item_index"
        ].apply(
            lambda idx: predict_svd(user_index, int(idx))
        )
    else:
        matched_products["Predicted_Rating"] = matched_products[
            "item_index"
        ].apply(
            lambda idx: predict_knn(user_index, int(idx))
        )

    matched_products = matched_products.sort_values(
        ["Predicted_Rating", "Rating_Count"],
        ascending=False
    )

    print("\nPersonalized recommendations using", best_model_name)

else:
    if user_input:
        print(
            "That user_id is not in the training data. "
            "Showing general product recommendations instead."
        )

    # For non-personalized recommendations, rank by observed ratings.
    # Require a minimum number of ratings to reduce one-rating noise.
    matched_products = matched_products[
        matched_products["Rating_Count"] >= 2
    ].copy()

    matched_products = matched_products.sort_values(
        ["Average_Rating", "Rating_Count"],
        ascending=False
    )

    matched_products["Predicted_Rating"] = (
        matched_products["Average_Rating"]
    )

    print("\nGeneral recommendations ranked by observed ratings.")

# Show the top ten recommendations
recommendations = matched_products.head(10).copy()

if recommendations.empty:
    print("No suitable recommendations could be generated.")
else:
    display_columns = [
        "item_id", "Brand", "Category",
        "Model_Attribute", "Average_Rating",
        "Rating_Count", "Predicted_Rating"
    ]

    print("\nTOP 10 RECOMMENDED PRODUCTS")
    print(
        recommendations[display_columns]
        .round(3)
        .to_string(index=False)
    )

    recommendations[display_columns].to_csv(
        "recommended_electronics_products.csv",
        index=False
    )

    # Brand-level summary of the recommended products
    brand_summary = (
        recommendations.groupby("Brand")
        .agg(
            Recommended_Products=("item_id", "count"),
            Mean_Predicted_Rating=("Predicted_Rating", "mean"),
            Mean_Observed_Rating=("Average_Rating", "mean")
        )
        .reset_index()
        .sort_values(
            ["Mean_Predicted_Rating", "Recommended_Products"],
            ascending=False
        )
    )

    print("\nRECOMMENDED BRANDS")
    print(brand_summary.round(3).to_string(index=False))

    if not brand_summary.empty:
        print(
            "\nTop brand among the displayed recommendations:",
            brand_summary.iloc[0]["Brand"]
        )

# ==========================================
# 13. FINAL CONCLUSION
# ==========================================

print("\n========== PROJECT CONCLUSION ==========")
print("Models Compared: KNN and SVD")
print("Best Model Based on F1-Score:", best_model_name)
print("Best F1-Score:", round(best_row["F1-Score"], 4))
print("Best Accuracy:", round(best_row["Accuracy"], 4))


print("\nPROJECT COMPLETED SUCCESSFULLY")

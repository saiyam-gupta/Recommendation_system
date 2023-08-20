import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise.accuracy import rmse


# Define the number of users and products
num_users = 10000
num_products = 10000
# Define product categories
product_categories = ["Electronics", "Clothing", "Home", "Books", "Beauty", "Sports", "Toys", "Food"]
# Create empty lists to store user data and interactions
users = []
products = []
interactions = []

# Simulate user profiles
for user_id in range(num_users):
    user = {
        "user_id": user_id,
        "age": random.randint(18, 70),
        "gender": random.choice(["Male", "Female", "Other"]),
        "preferences": random.sample(product_categories, random.randint(1, len(product_categories))),
        "location": random.choice(["North", "South", "East", "West"])
    }
    users.append(user)

# Simulate product data
for product_id in range(num_products):
    product = {
        "product_id": product_id,
        "category": random.choice(product_categories),
        "rating": random.randint(1, 5)
    }
    products.append(product)

# Simulate interactions
current_time = datetime.now()
for user in users:
    num_interactions = random.randint(0, 10)
    for _ in range(num_interactions):
        product = random.choice(products)
        interaction_type = random.choice(["click", "purchase","Other"])
        interaction = {
            "user_id": user["user_id"],
            "product_id": product["product_id"],
            "timestamps": [current_time - timedelta(days=random.randint(1, 365))],
            "interaction_type": interaction_type,
        }
        interactions.append(interaction)

# Create a DataFrame for interactions
interactions_df = pd.DataFrame(interactions)

# Convert interaction_type to binary values (0 for no interaction, 1 for interaction)
# interactions_df["interaction_type"] = interactions_df["interaction_type"].apply(lambda x: 1 if x == "purchase" else 0)
interactions_df["interaction_type"] = interactions_df["interaction_type"].apply(lambda x: 5 if x == "purchase" else 1 if x == "click" else 0)


# Create user-product interaction matrix (dense) with implicit feedback for training
train_interaction_matrix = np.zeros((num_users, num_products))
for interaction in interactions:
    user_id = interaction["user_id"]
    product_id = interaction["product_id"]
    train_interaction_matrix[user_id, product_id] += 1


# Surprise library uses a different data structure
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(interactions_df[["user_id", "product_id", "interaction_type"]], reader)
trainset, testset = surprise_train_test_split(data, test_size=0.2)


# Matrix factorization using Surprise's SVD
model = SVD(n_factors=50, reg_all=0.01)
trainset = data.build_full_trainset()  # Build the training set using the entire data
model.fit(trainset)

# Predict user-item interactions using SVD model
user_item_predictions = model.test(testset)

# Convert user-item predictions to a user-product interaction matrix
user_product_interaction_matrix = np.zeros((num_users, num_products))
for prediction in user_item_predictions:
    user_idx = prediction.uid
    product_idx = prediction.iid
    predicted_interaction = prediction.est
    user_product_interaction_matrix[user_idx, product_idx] = predicted_interaction

# Content-Based Filtering: Enhance user profiles with textual data
product_descriptions = [product["category"] for product in products]  # Use product category as a simple description
user_profiles_text = [" ".join(user["preferences"]) for user in users]
tfidf_vectorizer = TfidfVectorizer()
user_profiles_tfidf = tfidf_vectorizer.fit_transform(user_profiles_text)
product_descriptions_tfidf = tfidf_vectorizer.transform(product_descriptions)
content_based_scores = cosine_similarity(user_profiles_tfidf, product_descriptions_tfidf)

# Calculate User Similarity using Gower's similarity coefficient
user_attributes = np.array([(user["age"], user["gender"], user["location"]) for user in users])

# Convert categorical attributes to numerical values
gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
location_mapping = {"North": 0, "South": 1, "East": 2, "West": 3}

for user in user_attributes:
    user[1] = gender_mapping[user[1]]
    user[2] = location_mapping[user[2]]

# Calculate Gower's similarity coefficient
user_similarity_matrix = 1 - pairwise_distances(user_attributes, metric="cosine")

# Normalize the scores
user_product_interaction_matrix_normalized = user_product_interaction_matrix / np.max(user_product_interaction_matrix)
content_based_scores_normalized = content_based_scores / np.max(content_based_scores)
user_similarity_matrix_normalized = user_similarity_matrix / np.max(user_similarity_matrix)

# Combine Recommendations from Different Models
alpha = 0.5  # Weight for collaborative filtering
beta = 0.2   # Weight for content-based filtering
gamma = 0.2  # Weight for user similarity
delta = 0.1  # Weight for product ratings

# Combine the normalized scores using the specified weights
# Create weight matrices for the models
weights_cf = np.ones((num_users, num_products)) * alpha
weights_cb = np.ones((num_users, num_products)) * beta
weights_sim = np.ones((num_users, num_products)) * gamma
weights_rating = np.array([product["rating"] for product in products]) * delta

# Combine the scores using matrix multiplication
combined_scores = (
    weights_cf * user_product_interaction_matrix_normalized +
    weights_cb * content_based_scores_normalized +
    weights_sim * user_similarity_matrix_normalized +
    np.outer(np.ones(num_users), weights_rating)
)

# Normalize the combined scores
combined_scores_normalized = combined_scores / np.max(combined_scores)

# Evaluate the Model using Testing Dataset
def evaluate_model(testset, num_recommendations):
    total_rmse = 0
    total_users = len(set(uid for uid, _, _ in testset))  # Extract user IDs from testset

    for uid, _, _ in testset:
        user_idx = uid  # Assuming user indices match user IDs
        user_combined_scores = combined_scores_normalized[user_idx]
        top_recommendations = np.argsort(user_combined_scores)[::-1][:num_recommendations]

        actual_interactions = train_interaction_matrix[uid]
        predicted_interactions = user_combined_scores[top_recommendations]
        rmse = np.sqrt(np.mean((actual_interactions[top_recommendations] - predicted_interactions) ** 2))
        total_rmse += rmse

    mean_rmse = total_rmse / total_users
    return mean_rmse

# Evaluate the Model using Testing Dataset
num_recommendations = 5
mean_rmse = evaluate_model(testset, num_recommendations)
print(f"Mean RMSE for Predicted Interactions: {mean_rmse:.4f}")

# Print recommendations for a specific user
user_id_to_recommend = 5  # Change this to the user ID you want to recommend for
user_combined_scores = combined_scores_normalized[user_id_to_recommend]
top_recommendations = np.argsort(user_combined_scores)[::-1][:num_recommendations]

recommended_product_ids = [products[product_id]['product_id'] for product_id in top_recommendations]
print(f"Recommended Product IDs for User {user_id_to_recommend}: {recommended_product_ids}")

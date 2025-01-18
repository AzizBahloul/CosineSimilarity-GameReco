import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load your dataset
df = pd.read_csv('data\data.csv')
# Convert ratings like '5/10' to float
df['Personal Rating'] = df['Personal Rating'].str.replace('/10', '').astype(float)
# One-hot encode categorical features
categorical_features = ['Genre', 'Platform', 'Developer', 'Publisher']
encoder = OneHotEncoder()
categorical_encoded = encoder.fit_transform(df[categorical_features]).toarray()
# TF-IDF for descriptions
tfidf = TfidfVectorizer(stop_words='english')
description_tfidf = tfidf.fit_transform(df['Description']).toarray()
# Normalize numerical features
numerical_features = ['Graphics Quality', 'Difficulty', 'Replayability', 'Personal Rating']
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(df[numerical_features])
# Combine all features into a single feature matrix
features = np.hstack((categorical_encoded, description_tfidf, numerical_scaled))
# Use TSNE for a 2D visualization
X_embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(features)
similarity_matrix = cosine_similarity(X_embedded)

# Simple TSNE scatter plot
plt.figure(figsize=(8,6))
plt.scatter(X_embedded[:,0], X_embedded[:,1])
plt.title("Feature Space Visualization")
plt.show()

# Save the encoder, scaler, tfidf, and similarity matrix
pickle.dump(encoder, open('models\encoder.pkl', 'wb'))
pickle.dump(scaler, open('models\scaler.pkl', 'wb'))
pickle.dump(tfidf, open('models\ tfidf.pkl', 'wb'))
pickle.dump(similarity_matrix, open('models\similarity_matrix.pkl', 'wb'))

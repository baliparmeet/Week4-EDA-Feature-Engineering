
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned Sephora dataset
df = pd.read_csv("sephora_reviews_cleaned.csv")

# Display the first 5 rows
print("First 5 rows:")
print(df.head())

# Check the number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Check data types and missing values
print("\nDataset Information:")
df.info()

# --------------------------------
# STEP 4: STATISTICAL ANALYSIS
# --------------------------------

# 1. Statistical summary of numerical columns
print("\nStatistical Summary:")
print(df.describe())

# 2. Check rating distribution
print("\nRating Distribution:")
print(df['rating'].value_counts().sort_index())

# 3. Check recommendation distribution
print("\nRecommendation Distribution:")
print(df['is_recommended'].value_counts(dropna=False))

# 4. Average product price
print("\nAverage Product Price:")
print(df['price_usd'].mean())

# 5. Most reviewed brands
print("\nTop 10 Most Reviewed Brands:")
print(df['brand_name'].value_counts().head(10))
# Rating distribution
plt.figure(figsize=(8, 5))

sns.countplot(data=df, x='rating')

plt.title("Distribution of Customer Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")

plt.show()
# Top 10 brands
top_brands = df['brand_name'].value_counts().head(10)

plt.figure(figsize=(10, 6))

sns.barplot(x=top_brands.values, y=top_brands.index)

plt.title("Top 10 Most Reviewed Brands")
plt.xlabel("Number of Reviews")
plt.ylabel("Brand Name")

plt.show()
# Product price distribution
plt.figure(figsize=(8, 5))

sns.histplot(data=df, x='price_usd', bins=30)

plt.title("Distribution of Product Prices")
plt.xlabel("Price in USD")
plt.ylabel("Frequency")

plt.show()

# --------------------------------
# STEP 7: RELATIONSHIP ANALYSIS
# --------------------------------

# 1. Relationship between price and rating
plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x='rating', y='price_usd')

plt.title("Product Price Distribution by Rating")
plt.xlabel("Customer Rating")
plt.ylabel("Price (USD)")

plt.show()


# 2. Average rating by recommendation status
plt.figure(figsize=(8, 5))

sns.barplot(data=df, x='is_recommended', y='rating')

plt.title("Average Rating by Recommendation Status")
plt.xlabel("Recommended (0 = No, 1 = Yes)")
plt.ylabel("Average Rating")

plt.show()


# 3. Relationship between helpfulness and rating
plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x='rating', y='helpfulness')

plt.title("Review Helpfulness by Rating")
plt.xlabel("Customer Rating")
plt.ylabel("Helpfulness Score")

plt.show()

# --------------------------------
# STEP 9: ADVANCED EDA
# --------------------------------

# Convert submission_time into datetime
df['submission_time'] = pd.to_datetime(
    df['submission_time'],
    errors='coerce'
)

# Extract year
df['review_year'] = df['submission_time'].dt.year

print("\nReview Year Summary:")
print(df['review_year'].value_counts().sort_index())

# Skin type distribution

skin_counts = df['skin_type'].value_counts()

print("\nSkin Type Distribution:")
print(skin_counts)

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    y='skin_type',
    order=skin_counts.index
)

plt.title("Distribution of Customer Skin Types")
plt.xlabel("Number of Reviews")
plt.ylabel("Skin Type")

plt.show()

# Number of reviews by year

yearly_reviews = df['review_year'].value_counts().sort_index()

plt.figure(figsize=(10, 5))

sns.lineplot(
    x=yearly_reviews.index,
    y=yearly_reviews.values,
    marker='o'
)

plt.title("Customer Review Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Reviews")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Average rating of top 10 brands

top_10_brands = df['brand_name'].value_counts().head(10).index

brand_ratings = (
    df[df['brand_name'].isin(top_10_brands)]
    .groupby('brand_name')['rating']
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Rating by Top 10 Brands:")
print(brand_ratings)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=brand_ratings.values,
    y=brand_ratings.index
)

plt.title("Average Rating of Top 10 Most Reviewed Brands")
plt.xlabel("Average Rating")
plt.ylabel("Brand")

plt.show()

# ==========================================
# FEATURE ENGINEERING
# ==========================================

# 1. Calculate review length (number of words)
df['review_length'] = df['review_text_final'].fillna('').apply(
    lambda x: len(x.split())
)

# 2. Extract review year and month
df['submission_time'] = pd.to_datetime(
    df['submission_time'],
    format='%d-%m-%Y',
    errors='coerce'
)

df['review_year'] = df['submission_time'].dt.year
df['review_month'] = df['submission_time'].dt.month

# 3. Categorize product prices
df['price_category'] = pd.cut(
    df['price_usd'],
    bins=[-float('inf'), 28, 74, float('inf')],
    labels=['Budget', 'Mid-Range', 'Premium']
)

# 4. Categorize customer ratings
df['rating_category'] = pd.cut(
    df['rating'],
    bins=[0, 2, 3, 5],
    labels=['Negative', 'Neutral', 'Positive']
)

# Display the newly created features
print("\nNew Features:")
print(df[[
    'review_length',
    'review_year',
    'review_month',
    'price_category',
    'rating_category'
]].head())

print("\nUpdated Dataset Shape:")
print(df.shape)

print("\nFeature Engineering Completed!")

# ==========================================
# ANALYSIS OF ENGINEERED FEATURES
# ==========================================

# 1. Average review length by rating category
print("\nAverage Review Length by Rating:")
print(df.groupby('rating_category', observed=True)['review_length'].mean())

# 2. Number of products in each price category
print("\nProducts/Reviews by Price Category:")
print(df['price_category'].value_counts())

# 3. Average rating by price category
print("\nAverage Rating by Price Category:")
print(df.groupby('price_category', observed=True)['rating'].mean())

# 4. Number of reviews by month
print("\nReviews by Month:")
print(df['review_month'].value_counts().sort_index())

# 5. Average helpfulness by rating category
print("\nHelpfulness by Rating Category:")
print(df.groupby('rating_category', observed=True)['helpfulness'].mean())
# ----------------------------------
# NORMALIZATION
# ----------------------------------

# Normalize product price between 0 and 1
df['price_normalized'] = (
    (df['price_usd'] - df['price_usd'].min()) /
    (df['price_usd'].max() - df['price_usd'].min())
)

# Normalize review length between 0 and 1
df['review_length_normalized'] = (
    (df['review_length'] - df['review_length'].min()) /
    (df['review_length'].max() - df['review_length'].min())
)


# ----------------------------------
# ENCODING
# ----------------------------------

# Convert categorical features into numerical columns
encoded_features = pd.get_dummies(
    df[['price_category', 'rating_category']],
    prefix=['price', 'rating'],
    dtype=int
)

# Add encoded columns to the dataset
df = pd.concat([df, encoded_features], axis=1)


# ----------------------------------
# INTERACTION FEATURE
# ----------------------------------

# Combine product price and customer rating
df['price_rating_interaction'] = (
    df['price_usd'] * df['rating']
)


# ----------------------------------
# DISPLAY RESULTS
# ----------------------------------

print("\nNew Engineered Features:")
print(df[[
    'price_normalized',
    'review_length_normalized',
    'price_rating_interaction'
]].head())

print("\nEncoded Features:")
print(encoded_features.head())

print("\nFinal Dataset Shape:", df.shape)

print("\nAll Feature Engineering Steps Completed!")
# ----------------------------------
# SAVE FINAL DATASET
# ----------------------------------

df.to_csv("sephora_reviews_week4_final.csv", index=False)

df.to_excel("sephora_reviews_week4_final.xlsx", index=False)

print("\nFinal dataset saved successfully!")
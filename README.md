# import pandas as pd
import numpy as np

# Create the dataset (Tennis Play example)
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Function to calculate entropy
def calculate_entropy(df, target_col):
    entropy = 0
    values = df[target_col].unique()
    for value in values:
        probability = df[target_col].value_counts()[value] / len(df[target_col])
        entropy += -probability * np.log2(probability)
    return entropy

# Function to calculate information gain
def calculate_information_gain(df, feature, target_col):
    entropy_total = calculate_entropy(df, target_col)
    values = df[feature].unique()
    entropy_sum = 0

    for value in values:
        subset = df[df[feature] == value]
        entropy = calculate_entropy(subset, target_col)
        probability = len(subset) / len(df)
        entropy_sum += probability * entropy

    return entropy_total - entropy_sum

# ID3 Algorithm
def id3_algorithm(df, target_col, features):
    if len(df[target_col].unique()) == 1:
        return df[target_col].iloc[0]
    if len(features) == 0:
        return df[target_col].value_counts().idxmax()

    information_gain = [calculate_information_gain(df, feature, target_col) for feature in features]
    best_feature_index = np.argmax(information_gain)
    best_feature = features[best_feature_index]

    tree = {best_feature: {}}
    features = [f for f in features if f != best_feature]

    for value in df[best_feature].unique():
        subset = df[df[best_feature] == value]
        subtree = id3_algorithm(subset, target_col, features)
        tree[best_feature][value] = subtree

    return tree

# List of features (excluding the target column)
features = df.columns[:-1].tolist()

# Building the ID3 decision tree
decision_tree = id3_algorithm(df, 'PlayTennis', features)

# Display the resulting decision tree
print(decision_tree)

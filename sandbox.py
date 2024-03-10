from sklearn import datasets
import pandas as pd


iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(dir(iris))
target = iris.target
print(target.shape)

# Display the first few rows of the DataFrame
print(df.head())

# Get summary statistics of the DataFrame
print(df.describe())

# Count the number of occurrences of each class in the target variable
print(df['target'].value_counts())

# Plot a histogram of each feature
df.hist(figsize=(10, 6))

# Plot a scatter matrix to visualize the relationships between features
pd.plotting.scatter_matrix(df, figsize=(12, 8));
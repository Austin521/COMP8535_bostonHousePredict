import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# File path
file_path = 'data/boston.txt'

# Read file content
with open(file_path, 'r') as file:
    data = file.readlines()

# Process data
processed_data = []
temp_line = []

for line in data:
    # Clean data, remove extra spaces and split by space
    line_data = line.strip().split()
    # Skip non-data lines
    if not line_data or not all(item.replace('.', '', 1).replace('-', '', 1).isdigit() for item in line_data):
        continue
    temp_line.extend(line_data)
    # If length reaches 14, add to processed_data
    if len(temp_line) == 14:
        processed_data.append(temp_line)
        temp_line = []

# Convert to numpy array
data_matrix = np.array(processed_data, dtype=float)

# Convert data to DataFrame
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.DataFrame(data_matrix, columns=columns)

# Calculate correlation matrix and plot heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Plot pairplot
sns.pairplot(df)
plt.show()

# Calculate VIF values
X = df.drop(columns=['MEDV'])  # 'MEDV' is the dependent variable
X = sm.add_constant(X)

vif = pd.DataFrame()
vif['Variable'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)

# Remove the constant column for model training
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Apply Kernel PCA
kpca = KernelPCA(n_components=2, kernel='poly', degree=3)
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Define data types for comparison
data_types = {
    'Original': (X_train, X_test),
    'PCA': (X_train_pca, X_test_pca),
    'Kernel PCA': (X_train_kpca, X_test_kpca)
}

# Evaluate models
results = []
for model_name, model in models.items():
    for data_name, (X_tr, X_te) in data_types.items():
        model.fit(X_tr, y_train)
        y_pred_train = model.predict(X_tr)
        y_pred_test = model.predict(X_te)

        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        scores = cross_val_score(model, X_tr, y_train, cv=10, scoring='neg_mean_squared_error')
        cv_mse = -scores.mean()
        cv_r2 = cross_val_score(model, X_tr, y_train, cv=10, scoring='r2').mean()

        results.append({
            'Model': model_name,
            'Data Type': data_name,
            'Train MSE': mse_train,
            'Test MSE': mse_test,
            'CV MSE': cv_mse,
            'CV R-Squared': cv_r2
        })

# Convert to DataFrame and display results
results_df = pd.DataFrame(results)
print(results_df)

# Plot results table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.show()

# Calculate and plot feature importance
model = models['Random Forest']
model.fit(X_train, y_train)  # Fit the model again with the original data
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Ensure feature names match the importance
feature_names = X.columns  # Use original feature names

sorted_feature_names = [feature_names[i] for i in indices]
sorted_importances = importances[indices]

# Create a DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': sorted_feature_names,
    'Importance': sorted_importances
})

plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis", hue='Feature', dodge=False, legend=False)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Additional Analysis: PCA and Kernel PCA feature importance

# PCA Feature Importance
pca_components = pca.components_
pca_importance = np.abs(pca_components).sum(axis=0)
pca_feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'PCA Importance': pca_importance / pca_importance.sum()
})
pca_feature_importance = pca_feature_importance.sort_values(by='PCA Importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.title("PCA Feature Importance")
sns.barplot(x='PCA Importance', y='Feature', data=pca_feature_importance, hue='Feature', palette="viridis", dodge=False, legend=False)
plt.xlabel('PCA Importance')
plt.ylabel('Features')
plt.show()

# Kernel PCA Feature Importance
kpca_X_transformed = kpca.transform(X)
kpca_importance = np.var(kpca_X_transformed, axis=0)
kpca_feature_importance = pd.DataFrame({
    'Component': ['PC1', 'PC2'],
    'Kernel PCA Importance': kpca_importance / kpca_importance.sum()
})

plt.figure(figsize=(12, 6))
plt.title("Kernel PCA Component Importance")
sns.barplot(x='Kernel PCA Importance', y='Component', data=kpca_feature_importance, hue='Component', palette="viridis", dodge=False, legend=False)
plt.xlabel('Kernel PCA Importance')
plt.ylabel('Components')
plt.show()

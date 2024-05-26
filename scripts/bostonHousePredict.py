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
# 为PCA应用累积方差，找出需要多少个成分才能达到95%的方差解释
pca = PCA().fit(X_train)  # 先不限制成分数，以计算所有主成分的方差解释
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components_pca = np.where(cumulative_variance_ratio >= 0.95)[0][0] + 1  # 加1因为索引从0开始

print(f"Number of PCA components to retain 95% variance: {n_components_pca}")

# 使用所需数量的成分重新应用PCA
pca = PCA(n_components=n_components_pca)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Apply Kernel PCA
kpca = KernelPCA(n_components=n_components_pca, kernel='poly', degree=3)
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)

print(f"Number of Kernel PCA components to retain 95% variance: {n_components_pca}")

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

        # Plot scatter plot of actual vs predicted values for test data
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, edgecolor='k', alpha=0.7, s=70)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{model_name} with {data_name} Data')
        plt.show()

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

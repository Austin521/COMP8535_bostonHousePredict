import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE

# Define the correct file path
file_path = "/Users/austinlee/Desktop/pythonProject/data/boston.txt"

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

# Split data into features and target
X = data_matrix[:, :-1]
y = data_matrix[:, -1]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tsne, y, test_size=0.2, random_state=42)

# Random Forest parameter tuning
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# Evaluate best Random Forest model
best_rf_model = rf_grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f'Best Random Forest Test MSE: {mse_rf}')

cv_scores_rf = cross_val_score(best_rf_model, X_tsne, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
cv_mse_rf = -np.mean(cv_scores_rf)
print(f'Best Random Forest Cross-Validation MSE: {cv_mse_rf}')

# Gradient Boosting parameter tuning
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.05],
    'subsample': [0.8, 0.9, 1.0],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb_grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), param_grid=gb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gb_grid_search.fit(X_train, y_train)

# Evaluate best Gradient Boosting model
best_gb_model = gb_grid_search.best_estimator_
y_pred_gb = best_gb_model.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
print(f'Best Gradient Boosting Test MSE: {mse_gb}')

cv_scores_gb = cross_val_score(best_gb_model, X_tsne, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
cv_mse_gb = -np.mean(cv_scores_gb)
print(f'Best Gradient Boosting Cross-Validation MSE: {cv_mse_gb}')

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

# 文件路径
file_path = 'data/boston.txt'

# 读取文件内容
with open(file_path, 'r') as file:
    data = file.readlines()

# 处理数据
processed_data = []
temp_line = []

for line in data:
    # 清理数据，去掉多余的空格并按空格分割
    line_data = line.strip().split()
    # 跳过非数据行
    if not line_data or not all(item.replace('.', '', 1).replace('-', '', 1).isdigit() for item in line_data):
        continue
    temp_line.extend(line_data)
    # 如果长度达到14，则添加到processed_data
    if len(temp_line) == 14:
        processed_data.append(temp_line)
        temp_line = []

# 转换为numpy数组
data_matrix = np.array(processed_data, dtype=float)

# 使用SimpleImputer来处理缺失值，这里我们使用均值来填补缺失值
imputer = SimpleImputer(strategy='mean')
data_matrix = imputer.fit_transform(data_matrix)

# 打印数据矩阵，设置打印选项为常规小数表示
np.set_printoptions(suppress=True, precision=4)
print("数据矩阵 (前五行)：")
print(data_matrix[:5])

# 提取特征和目标变量（确保前13列是特征，最后一列是目标变量）
X = data_matrix[:, :-1]  # 前13列是特征变量
y = data_matrix[:, -1]   # 最后一列是目标变量

# 使用KernelPCA计算累积解释方差比例，使用多项式核
kpca = KernelPCA(n_components=X.shape[1], kernel='poly', degree=3, coef0=1)
X_kpca_full = kpca.fit_transform(X)
explained_variance = np.var(X_kpca_full, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# 打印每个主成分的解释方差比例
print("每个主成分的解释方差比例：")
for i, var_ratio in enumerate(explained_variance_ratio):
    print(f"主成分 {i+1}: {var_ratio:.4f}")

# 绘制累积解释方差比例图
plt.figure(figsize=(8, 5))
plt.plot(range(1, X.shape[1] + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Principal Components')
plt.grid()
plt.show()

# 确定保留的主成分数量，例如累积解释方差超过95%的点
n_components = np.argmax(cumulative_explained_variance >= 0.95) + 1
print(f"保留的主成分数量: {n_components}")

# 使用KernelPCA降维
kpca = KernelPCA(n_components=n_components, kernel='poly', degree=3, coef0=1)
X_kpca = kpca.fit_transform(X)

# 交叉验证设置
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 初始化线性回归模型
lin_reg = LinearRegression()

# 线性回归交叉验证
cv_mse_lin = cross_val_score(lin_reg, X_kpca, y, cv=kf, scoring='neg_mean_squared_error')
cv_r2_lin = cross_val_score(lin_reg, X_kpca, y, cv=kf, scoring='r2')

print('线性回归交叉验证结果:')
print(f'均方误差 (Cross-Validation MSE): {-np.mean(cv_mse_lin)}')
print(f'R^2得分 (Cross-Validation R^2): {np.mean(cv_r2_lin)}')

# 初始化随机森林回归模型
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# 随机森林回归交叉验证
cv_mse_rf = cross_val_score(rf_reg, X_kpca, y, cv=kf, scoring='neg_mean_squared_error')
cv_r2_rf = cross_val_score(rf_reg, X_kpca, y, cv=kf, scoring='r2')

print('\n随机森林回归交叉验证结果:')
print(f'均方误差 (Cross-Validation MSE): {-np.mean(cv_mse_rf)}')
print(f'R^2得分 (Cross-Validation R^2): {np.mean(cv_r2_rf)}')

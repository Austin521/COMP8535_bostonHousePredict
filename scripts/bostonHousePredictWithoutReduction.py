import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

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

# 使用PCA降维
pca = PCA(n_components=10)  # 假设我们降到10个主成分
X_pca = pca.fit_transform(X)

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 初始化线性回归模型
lin_reg = LinearRegression()

# 训练线性回归模型
lin_reg.fit(X_train, y_train)

# 线性回归预测
y_pred_lin = lin_reg.predict(X_test)

# 线性回归评估
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

print('线性回归结果:')
print(f'均方误差: {mse_lin}')
print(f'R^2得分: {r2_lin}')
print('回归系数:', lin_reg.coef_)
print('截距:', lin_reg.intercept_)

# 初始化随机森林回归模型
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练随机森林回归模型
rf_reg.fit(X_train, y_train)

# 随机森林回归预测
y_pred_rf = rf_reg.predict(X_test)

# 随机森林回归评估
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print('\n随机森林回归结果:')
print(f'均方误差: {mse_rf}')
print(f'R^2得分: {r2_rf}')

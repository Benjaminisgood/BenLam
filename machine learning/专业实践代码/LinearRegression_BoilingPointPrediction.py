# 导入必要的库
import pandas as pd  # 数据处理库
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.model_selection import train_test_split  # 拆分训练集和测试集
import matplotlib.pyplot as plt  # 可视化绘图库（只用于英文标签）

# ─── 构造示例数据 ─────────────────────────────
# 包含了5个有机分子的特征与沸点数据
data = {
    "MolecularWeight": [18, 46, 60, 74, 88],  # 分子量
    "HBD": [1, 1, 2, 1, 1],  # 氢键供体数
    "HBA": [2, 2, 2, 2, 2],  # 氢键受体数
    "TPSA": [20.2, 46.5, 46.5, 46.5, 46.5],  # 极性表面积
    "BoilingPoint": [100, 78, 82, 97, 117]  # 沸点（℃）
}
df = pd.DataFrame(data)  # 转换为DataFrame格式便于处理

# ─── 准备特征（X）和标签（y） ─────────────────
X = df[["MolecularWeight", "HBD", "HBA", "TPSA"]]  # 模型输入特征
y = df["BoilingPoint"]  # 模型要预测的目标

# ─── 拆分训练集和测试集（80%训练，20%测试） ─────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ─── 建立并训练线性回归模型 ─────────────────────
model = LinearRegression()  # 初始化模型
model.fit(X_train, y_train)  # 在训练集上拟合模型

# ─── 使用测试集进行预测 ─────────────────────────
y_pred = model.predict(X_test)  # 得到预测结果

# ─── 画出预测值与真实值的对比图 ─────────────────
plt.scatter(y_test, y_pred)  # 散点图：横坐标是真实值，纵坐标是预测值
plt.xlabel("True Boiling Point")  # 英文坐标标签
plt.ylabel("Predicted Boiling Point")
plt.title("Prediction vs True Value")  # 图标题
plt.plot([70, 120], [70, 120], '--', color='gray')  # 灰色对角参考线（理想预测）
plt.grid(True)  # 显示网格线
plt.show()  # 显示图像

# 假设模型已经训练好了（你之前已经fit过）
# 下面是你新找到的一个分子的性质：
new_data = {
    "MolecularWeight": [100],
    "HBD": [1],
    "HBA": [2],
    "TPSA": [46.5]
}

# 将其转换为DataFrame格式（模型要求的输入格式）
new_df = pd.DataFrame(new_data)

# 使用模型进行预测
predicted_bp = model.predict(new_df)

# 输出预测结果
print("Predicted Boiling Point:", predicted_bp[0])

input("Press Enter to close the figure and end the script.")
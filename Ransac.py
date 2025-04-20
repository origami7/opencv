import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import Csv


# 使用 RANSAC 拟合直线
def fit_line_ransac(points, residual_threshold=1.0, max_trials=100, min_samples=10):
    if len(points) < min_samples:
        raise ValueError(
            f"Not enough points to fit a line. Required at least {min_samples} points, but got {len(points)} points.")

    X = points[:, 0].reshape(-1, 1)  # x 坐标
    y = points[:, 1]  # y 坐标

    # 创建 RANSAC 回归器，设置最小样本数为 10
    ransac = RANSACRegressor(estimator=LinearRegression(),
                             residual_threshold=residual_threshold,
                             max_trials=max_trials,
                             min_samples=min_samples)
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    return slope, intercept, inlier_mask


# 提取多条直线
def extract_multiple_lines(points, num_lines=3, residual_threshold=1.0, max_trials=100, min_samples=10):
    lines = []
    remaining_points = points.copy()

    for _ in range(num_lines):
        if len(remaining_points) < min_samples:
            print(f"Not enough points to fit another line. Stopping at {len(lines)} lines.")
            break

        try:
            slope, intercept, inlier_mask = fit_line_ransac(remaining_points, residual_threshold, max_trials,
                                                            min_samples)
            inlier_points = remaining_points[inlier_mask]
            lines.append((slope, intercept, inlier_points))
            remaining_points = remaining_points[~inlier_mask]
        except ValueError as e:
            print(e)
            break

    return lines


# 获取数据
points_1 = Test.filtered_points_at_z_1_2d
x, y = points_1[:, 0], points_1[:, 1]
max_y_index = np.argmax(y)
max_y_x_value = x[max_y_index]
point1 = points_1[points_1[:, 0] > max_y_x_value]

# 调整参数
residual_threshold = 0.05  # 残差阈值
max_trials = 1000  # 最大尝试次数
min_samples = 10  # 最小样本数

# 提取多条直线并输出方程
lines = extract_multiple_lines(point1, num_lines=3, residual_threshold=residual_threshold, max_trials=max_trials,
                               min_samples=min_samples)

# 输出直线方程
for i, (slope, intercept, inlier_points) in enumerate(lines):
    print(f"Line {i + 1}: y = {slope:.2f}x + {intercept:.2f}")

# 可视化结果
plt.figure(figsize=(8, 6))
for slope, intercept, inlier_points in lines:
    line_x = np.array([inlier_points[:, 0].min(), inlier_points[:, 0].max()])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color='red', linewidth=2)
    plt.scatter(inlier_points[:, 0], inlier_points[:, 1], color='blue')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('RANSAC Line Fitting')
plt.grid(True)
plt.show()
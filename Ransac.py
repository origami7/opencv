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


def find_parallel_line_with_larger_intercept_and_intersection(lines):
    # 初始化变量，用于存储最接近平行的两条直线及其斜率差
    min_slope_diff = float('inf')  # 初始化为无穷大
    parallel_lines = None

    # 遍历所有直线对，比较斜率差
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            slope_diff = abs(lines[i][0] - lines[j][0])
            if slope_diff < min_slope_diff:
                min_slope_diff = slope_diff
                parallel_lines = (lines[i], lines[j])

    # 找出截距较大的直线
    if parallel_lines:
        line1, line2 = parallel_lines
        if line1[1] > line2[1]:
            result_line = line1
        else:
            result_line = line2
    else:
        raise ValueError("没有找到接近平行的直线")

    # 找出剩下那条直线
    remaining_lines = [line for line in lines if line != line1 and line != line2]
    if not remaining_lines:
        raise ValueError("没有找到第三条直线")
    remaining_line = remaining_lines[0]

    # 计算交点
    def find_intersection(line1, line3):
        m1, b1 = line1[0], line1[1]
        m3, b3 = line3[0], line3[1]
        x = (b3 - b1) / (m1 - m3)
        y = m1 * x + b1
        return (x, y)

    intersection_point = find_intersection(result_line, remaining_line)

    return result_line, intersection_point


# 调整参数
residual_threshold = 0.05  # 残差阈值
max_trials = 1000  # 最大尝试次数
min_samples = 10  # 最小样本数
character_points = {}
for i, point1 in enumerate(Csv.Eight_Edge_list):
    # 提取多条直线并输出方程
    lines = extract_multiple_lines(point1, num_lines=3, residual_threshold=residual_threshold, max_trials=max_trials,
                                   min_samples=min_samples)

    # 输出直线方程
    for j, (slope, intercept, inlier_points) in enumerate(lines):
        print(f"Line {j + 1}: y = {slope:.2f}x + {intercept:.2f}")

    result_line, intersection_point = find_parallel_line_with_larger_intercept_and_intersection(lines)
    print(f"最接近平行的两条直线中截距较大的直线是：")
    print(f"斜率：{result_line[0]:.2f}, 截距：{result_line[1]:.2f}")
    print(f"该直线与剩下那条直线的交点坐标为：({intersection_point[0]:.2f}, {intersection_point[1]:.2f})")

    # 计算新点的坐标
    slope, intercept = result_line[0], result_line[1]
    if i % 2 == 0:  # i 为偶数
        distance = 2
    else:  # i 为奇数
        distance = -2

    # 计算新点的坐标
    x_new = intersection_point[0] + distance / np.sqrt(1 + slope**2)
    y_new = slope * x_new + intercept

    character_points[i+1] = [(intersection_point[0],intersection_point[1]),(x_new,y_new)]
    print(character_points)
    # 可视化结果
    plt.figure(figsize=(8, 6))
    for slope, intercept, inlier_points in lines:
        line_x = np.array([inlier_points[:, 0].min(), inlier_points[:, 0].max()])
        line_y = slope * line_x + intercept
        plt.plot(line_x, line_y, color='red', linewidth=2)
        plt.scatter(inlier_points[:, 0], inlier_points[:, 1], color='blue')

    # 绘制交点
    plt.scatter(intersection_point[0], intersection_point[1], color='green', marker='o', s=100, label='Intersection Point')

    # 绘制新点
    plt.scatter(x_new, y_new, color='green', marker='o', s=100, label='New Point')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'RANSAC Line Fitting with Intersection and New Point (Iteration {i + 1})')
    plt.legend()
    plt.grid(True)
    plt.show()
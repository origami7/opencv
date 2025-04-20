import pandas as pd
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# 定义一个函数，用于读取单个 CSV 文件并提取点云数据
def read_csv_and_extract_points(file_path):
    # 读取 CSV 文件，跳过前两行（假设前两行是标题或说明等非数据内容）
    df = pd.read_csv(file_path, skiprows=2, header=None)

    # 初始化一个空的列表，用于存储提取的点云数据
    points = []

    # 遍历每一行数据（每一行是一条轮廓线）
    for index, row in df.iterrows():
        # 获取当前行的数据
        data = row.values

        # 从第三列开始，每三个值为一组，分别代表 W、H、M
        for i in range(2, len(data), 3):
            # 提取 W、H、M 值
            W = data[i]
            H = data[i + 1]
            M = data[i + 2]

            # 将 W、H、M 转换为 x、y、z 坐标
            x = W
            y = H
            z = M

            # 检查是否为 (0, 0, 0) 点，如果不是，则添加到点云数据中
            if not (x == 0 and y == 0 and z == 0):
                points.append([x, y, z])

    # 将点云数据转换为 NumPy 数组
    return np.array(points)

# 将点云数据转换为 Open3D 的点云对象
def points_to_pointcloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

# 统计滤波函数
def statistical_filtering(points, nb_neighbors=20, std_ratio=2.0):
    point_cloud = points_to_pointcloud(points)
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(cl.points)

# 筛选接近特定 z 高度的点云数据
def filter_points_by_z(points, z_target, tolerance):
    z_diff = np.abs(points[:, 2] - z_target)
    return points[z_diff <= tolerance]

# 可视化点云数据
def visualize_point_cloud(points_list, colors_list, title="Point Cloud"):
    point_clouds = [points_to_pointcloud(points) for points in points_list]
    for point_cloud, color in zip(point_clouds, colors_list):
        point_cloud.paint_uniform_color(color)
    o3d.visualization.draw_geometries(point_clouds, window_name=title)

# 分离小于和大于指定 x 值的数据
def split_points_by_x(points):
    x, y = points[:, 0], points[:, 1]
    max_y_index = np.argmax(y)
    max_y_x_value = x[max_y_index]
    less_than_x = points[points[:, 0] < max_y_x_value]
    greater_than_x = points[points[:, 0] > max_y_x_value]
    return less_than_x, greater_than_x

# 将点云数据映射到 xy 平面上并使用 matplotlib 可视化，每组数据单独显示
def visualize_points_on_xy_plane(points_list, colors_list, title_prefix="XY Plane Projection"):
    for i, (points, color) in enumerate(zip(points_list, colors_list)):
        # 分离小于和大于 max_y_x_value 的数据
        less_than_x, greater_than_x = split_points_by_x(points)

        # 创建一个包含两个子图的图窗口
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # 绘制小于 max_y_x_value 的数据
        if len(less_than_x) > 0:
            axs[0].scatter(less_than_x[:, 0], less_than_x[:, 1], c=color, label=f"l_Sensor {i + 1}")
        axs[0].set_title("left_points")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_aspect("equal")

        # 绘制大于 max_y_x_value 的数据
        if len(greater_than_x) > 0:
            axs[1].scatter(greater_than_x[:, 0], greater_than_x[:, 1], c=color, label=f"r_Sensor {i + 1}")
        axs[1].set_title("right_points")
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_aspect("equal")

        plt.tight_layout()
        plt.show()

# 读取四个 CSV 文件并提取点云数据
csv_files = ["data/物料验证2/量块/左上.csv", "data/物料验证2/量块/右上.csv", "data/物料验证2/量块/右下.csv", "data/物料验证2/量块/左下.csv"]
sensor_points_1 = read_csv_and_extract_points(csv_files[0])
sensor_points_2 = read_csv_and_extract_points(csv_files[1])
sensor_points_3 = read_csv_and_extract_points(csv_files[2])
sensor_points_4 = read_csv_and_extract_points(csv_files[3])

# 统一高度2与其他一致
z_offset = np.mean(sensor_points_1[:, 2]) - np.mean(sensor_points_2[:, 2])
sensor_points_2[:, 2] += z_offset

# 对四个传感器的点云数据进行统计滤波
filtered_points_1 = statistical_filtering(sensor_points_1)
filtered_points_2 = statistical_filtering(sensor_points_2)
filtered_points_3 = statistical_filtering(sensor_points_3)
filtered_points_4 = statistical_filtering(sensor_points_4)

# 可视化四个传感器的所有点云数据
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]  # 红、绿、蓝、黄
#visualize_point_cloud([filtered_points_1, filtered_points_2, filtered_points_3, filtered_points_4], colors, title="Combined Filtered Points")

# 筛选接近目标 z 高度的点云数据
z_target = 3  # 根据需要调整z高度
tolerance = 0.03  # 容差范围，用于筛选接近 z_target 的点
filtered_points_at_z_1 = filter_points_by_z(filtered_points_1, z_target, tolerance)
filtered_points_at_z_2 = filter_points_by_z(filtered_points_2, z_target, tolerance)
filtered_points_at_z_3 = filter_points_by_z(filtered_points_3, z_target, tolerance)
filtered_points_at_z_4 = filter_points_by_z(filtered_points_4, z_target, tolerance)
#visualize_point_cloud([filtered_points_at_z_1, filtered_points_at_z_2, filtered_points_at_z_3, filtered_points_at_z_4], colors, title="Z Filtered Points")

# 可视化筛选 z 后的点云数据在 xy 平面上的投影，每组数据分开显示
filtered_points_at_z_1_2d = filtered_points_at_z_1[:, :2]
filtered_points_at_z_2_2d = filtered_points_at_z_2[:, :2]
filtered_points_at_z_3_2d = filtered_points_at_z_3[:, :2]
filtered_points_at_z_4_2d = filtered_points_at_z_4[:, :2]
colors_matplot = ['red', 'green', 'blue', 'yellow']  # 红、绿、蓝、黄
#visualize_points_on_xy_plane([filtered_points_at_z_1_2d, filtered_points_at_z_2_2d, filtered_points_at_z_3_2d, filtered_points_at_z_4_2d], colors_matplot, title_prefix="Points at z = {:.2f} on XY Plane".format(z_target))
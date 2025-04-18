import pandas as pd
import numpy as np
import open3d as o3d


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
    points = np.array(points)

    return points


# 将点云数据转换为 Open3D 的点云对象
def points_to_pointcloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


# 统计滤波函数
def statistical_filtering(points, nb_neighbors=20, std_ratio=2.0):
    point_cloud = points_to_pointcloud(points)
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    filtered_points = np.asarray(cl.points)
    return filtered_points

# 筛选接近特定 z 高度的点云数据
def filter_points_by_z(points, z_target, tolerance):
    # 计算每个点的 z 坐标与目标 z 高度的差值
    z_diff = np.abs(points[:, 2] - z_target)
    # 筛选出差值小于容差的点
    filtered_points = points[z_diff <= tolerance]
    return filtered_points

# 可视化点云数据
def visualize_point_cloud(points_list, colors_list, title="Point Cloud"):
    point_clouds = []
    for points, color in zip(points_list, colors_list):
        point_cloud = points_to_pointcloud(points)
        point_cloud.paint_uniform_color(color)
        point_clouds.append(point_cloud)

    o3d.visualization.draw_geometries(point_clouds, window_name=title)


# 定义四个数组，分别存储每个传感器的点云数据
sensor1_points = None
sensor2_points = None
sensor3_points = None
sensor4_points = None

# 读取四个 CSV 文件并提取点云数据
csv_files = ["data/物料验证2/量块/左上.csv", "data/物料验证2/量块/右上.csv", "data/物料验证2/量块/右下.csv",
             "data/物料验证2/量块/左下.csv"]

for i, file in enumerate(csv_files):
    points = read_csv_and_extract_points(file)
    if i == 0:
        sensor1_points = points
    elif i == 1:
        sensor2_points = points
    elif i == 2:
        sensor3_points = points
    elif i == 3:
        sensor4_points = points

#统一高度2与其他一致
z_offset = np.mean(sensor1_points[:, 2]) - np.mean(sensor2_points[:, 2])
sensor2_points[:, 2] += z_offset

# 对四个传感器的点云数据进行统计滤波
sensor1_points_filtered = statistical_filtering(sensor1_points)
sensor2_points_filtered = statistical_filtering(sensor2_points)
sensor3_points_filtered = statistical_filtering(sensor3_points)
sensor4_points_filtered = statistical_filtering(sensor4_points)

# 可视化四个传感器的所有点云数据
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]  # 红、绿、蓝、黄
points_list = [sensor1_points_filtered, sensor2_points_filtered, sensor3_points_filtered, sensor4_points_filtered]
visualize_point_cloud(points_list, colors, title="Combined Filtered Points")

z_target = 5  # 根据需要调整z高度
tolerance = 0.05  # 容差范围，用于筛选接近 z_target 的点

# 筛选接近目标 z 高度的点云数据
sensor1_points_filtered_at_z = filter_points_by_z(sensor1_points_filtered, z_target, tolerance)
sensor2_points_filtered_at_z = filter_points_by_z(sensor2_points_filtered, z_target, tolerance)
sensor3_points_filtered_at_z = filter_points_by_z(sensor3_points_filtered, z_target, tolerance)
sensor4_points_filtered_at_z = filter_points_by_z(sensor4_points_filtered, z_target, tolerance)

# 可视化筛选z后的点云数据
points_list_at_z = [sensor1_points_filtered_at_z, sensor2_points_filtered_at_z, sensor3_points_filtered_at_z, sensor4_points_filtered_at_z]
visualize_point_cloud(points_list_at_z, colors, title="Points at z = {:.2f}".format(z_target))
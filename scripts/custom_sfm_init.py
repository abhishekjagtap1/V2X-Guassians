import numpy as np
import open3d as o3d
import cv2

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

def transform_point_cloud(point_cloud, transformation_matrix):
    point_cloud.transform(transformation_matrix)
    return point_cloud

def backproject_depth_to_point_cloud(depth_image, rgb_image, intrinsic_matrix, extrinsic_matrix):
    height, width= depth_image.shape

    # Create mesh grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.flatten().astype(int)
    y = y.flatten().astype(int)
    z = depth_image.flatten()

    # Mask invalid depths
    mask = z > 0
    x = x[mask]
    y = y[mask]
    z = z[mask]

    # Convert to camera coordinates
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    points = np.stack((x, y, z), axis=-1)

    # Convert to homogeneous coordinates
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))

    # Transform to world coordinates using extrinsic matrix
    points_world = points_hom @ extrinsic_matrix.T
    points_world = points_world[:, :3]


    # Get RGB values
    x = x.astype(int)
    y = y.astype(int)
    valid_indices = (x >= 0) & (x < rgb_image.shape[1]) & (y >= 0) & (y < rgb_image.shape[0])
    x = x[valid_indices]
    y = y[valid_indices]
    points_world = points_world[valid_indices]
    rgb_values = rgb_image[y, x] / 255.0

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_world)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb_values)

    return point_cloud

def merge_point_clouds(point_clouds):
    merged_point_cloud = o3d.geometry.PointCloud()
    for pc in point_clouds:
        merged_point_cloud += pc
    merged_point_cloud = merged_point_cloud.voxel_down_sample(voxel_size=0.05)
    return merged_point_cloud

# Example file paths and matrices (to be replaced with actual data)
lidar_pcd_file = "/home/uchihadj/4DGaussians/data/lidar_initilization/south_vehicle_lidar_scene7.pcd"

depth_image_files = [
    "/home/uchihadj/TUMtraf/robust-dynrf/dataset/custom/Overlapping_scene_South_1and_2/south_2/dpt_png/1688625741_148698817_s110_camera_basler_south2_8mm.png",
    "/home/uchihadj/TUMtraf/robust-dynrf/dataset/custom/Overlapping_scene_South_1and_2/south_1/dpt_png/1688625741_172984730_s110_camera_basler_south1_8mm.png",


]

rgb_image_files = [
    "/home/uchihadj/TUMtraf/robust-dynrf/dataset/custom/Overlapping_scene_South_1and_2/south_2/images/1688625741_148698817_s110_camera_basler_south2_8mm.jpg",
    "/home/uchihadj/TUMtraf/robust-dynrf/dataset/custom/Overlapping_scene_South_1and_2/south_1/images/1688625741_172984730_s110_camera_basler_south1_8mm.jpg",

]

# Example intrinsic matrix
intrinsic_matrix = [np.array(
                            [[1315.56, 0, 969.353, 0.0], [0, 1368.35, 579.071, 0.0], [0, 0, 1, 0.0]],
                            dtype=float), np.array([[1400.3096617691212, 0.0, 967.7899705163408],
                                                              [0.0, 1403.041082755918, 581.7195041357244],
                                                              [0, 0, 1]], dtype=float)]



# Example extrinsic matrices for 4 cameras
transformation_matrix_lidar_to_base_south_2 = np.array([
    [0.247006, -0.955779, -0.15961, -16.8017],
    [0.912112, 0.173713, 0.371316, 4.66979],
    [-0.327169, -0.237299, 0.914685, 6.4602],
    [0.0, 0.0, 0.0, 1.0], ], dtype=float)

transformation_matrix_base_to_camera_south_2 = np.array([
    [0.8924758822566284, 0.45096261644035174, -0.01093243630327495, 14.921784677055939],
    [0.29913535165414396, -0.6097951995429897, -0.7339399539506467, 13.668310799382738],
    [-0.3376460291207414, 0.6517534297474759, -0.679126369559744, -5.630430017833277],
    [0.0, 0.0, 0.0, 1.0]], dtype=float)

extrinsic_matrix_lidar_to_camera_south_2 = np.matmul(
    transformation_matrix_base_to_camera_south_2,
    transformation_matrix_lidar_to_base_south_2)
#extrinsic_matrices = [np.linalg.inv(extrinsic_matrix_lidar_to_camera_south_2)]
#print(extrinsic_matrices.shape)
transformation_matrix_base_to_camera_south_1 = np.array([
    [0.891382638626301, 0.37756862104528707, -0.07884507325924934, 25.921784677055939],
    [0.2980421080238165, -0.6831891949380544, -0.6660273169946723, 13.668310799382738],
    [-0.24839844089507856, 0.5907739097931769, -0.7525203649548087, 18.630430017833277],
    [0, 0, 0, 1]], dtype=float)
transformation_matrix_lidar_to_base_south_1 = np.array([
    [0.247006, -0.955779, -0.15961, -16.8017],
    [0.912112, 0.173713, 0.371316, 4.66979],
    [-0.327169, -0.237299, 0.914685, 6.4602],
    [0.0, 0.0, 0.0, 1.0], ], dtype=float)

extrinsic_matrix_lidar_to_camera_south_1 = np.matmul(
    transformation_matrix_base_to_camera_south_1,
    transformation_matrix_lidar_to_base_south_1)
camera_to_lidar_extrinsics_south_1 = np.linalg.inv(extrinsic_matrix_lidar_to_camera_south_1)
#camera_to_lidar_extrinsics_south_1 = extrinsic_matrix_lidar_to_camera_south_1
dustr_poses = np.load(
    "/home/uchihadj/CVPR_2025/4DGaussians/data/4_agents_perfectly_alligned_pointcloud_poses/meta_data/4_agents_cam2_world.npy")
extrinsic_matrices = [np.linalg.inv(dustr_poses[0]), np.linalg.inv(dustr_poses[0])]
dustr_intrinsics = np.load("/home/uchihadj/CVPR_2025/4DGaussians/data/4_agents_perfectly_alligned_pointcloud_poses/meta_data/intrinsics_all.npy")
intrinsic_matrix = [dustr_intrinsics[0], dustr_intrinsics[3]]





# Load LiDAR point cloud
lidar_point_cloud = load_point_cloud(lidar_pcd_file)

# Load depth and RGB images for 4 cameras
depth_images = [load_image(file) for file in depth_image_files]
rgb_images = [load_image(file) for file in rgb_image_files]

# Generate point clouds from depth images
point_clouds = []
for i in range(2):
    point_cloud = backproject_depth_to_point_cloud(depth_images[i], rgb_images[i], intrinsic_matrix[i], extrinsic_matrices[i])
    point_clouds.append(point_cloud)

# Merge all point clouds
merged_point_cloud = merge_point_clouds(point_clouds)

# Visualize point cloud
o3d.visualization.draw_geometries([merged_point_cloud])

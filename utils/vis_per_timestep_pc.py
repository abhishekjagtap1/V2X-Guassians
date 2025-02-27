
import open3d as o3d
import open3d as o3d
import numpy as np

# Load the point cloud from the .ply file
#point_cloud = o3d.io.read_point_cloud("/home/uchihadj/CVPR_2025/4DGaussians/data/less_images_meta_data/point_cloud.ply")
#point_cloud = o3d.io.read_point_cloud("/home/uchihadj/CVPR_2025/4DGaussians/data/4_agents_perfectly_alligned_pointcloud_poses/meta_data/point_cloud.ply")
#point_cloud = o3d.io.read_point_cloud("/home/uchihadj/CVPR_2025/4DGaussians/data/4_agents_perfectly_alligned_pointcloud_poses/meta_data/downsampled_point_cloud.ply")

point_cloud = o3d.io.read_point_cloud("/home/uchihadj/CVPR_2025/dust3r/mesh_point_cloud.ply")

o3d.visualization.draw_geometries([point_cloud])

# Check if the point cloud has color information
#normals = np.array(point_cloud.opacties)
#print(normals.shape)
#print(n.shape)
if point_cloud.has_points():
    points = np.asarray(point_cloud.points)
    print(points.shape)
if point_cloud.has_colors():
    # Extract the colors to check their range
    colors = np.asarray(point_cloud.colors)
    print(colors.shape)
    print("Color range before normalization:", colors.min(), colors.max())

    # Normalize colors if they are in the range [0, 255]
    if colors.max() > 1.0:
        point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)
        print("Color range after normalization:", np.asarray(point_cloud.colors).min(),
              np.asarray(point_cloud.colors).max())

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud], window_name="Colored 3D Point Cloud", width=800, height=600)

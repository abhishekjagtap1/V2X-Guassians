import open3d as o3d


def crop_point_cloud(point_cloud, min_bound, max_bound):
    """
    Crop the point cloud to specified bounds.

    Args:
    - point_cloud (open3d.geometry.PointCloud): The input point cloud.
    - min_bound (tuple of float): The minimum x, y, z values.
    - max_bound (tuple of float): The maximum x, y, z values.

    Returns:
    - open3d.geometry.PointCloud: The cropped point cloud.
    """
    # Define the bounding box
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

    # Crop the point cloud
    cropped_point_cloud = point_cloud.crop(bounding_box)

    return cropped_point_cloud


# Load the point cloud from the .ply file
input_file = "/home/uchihadj/CVPR_2025/4DGaussians/utils/abbhi_utils/south_2_vehicle_scene_non_overlap_sientangled.ply"
output_file = "cropped_output.ply"

point_cloud = o3d.io.read_point_cloud(input_file)

# Get the bounds of the point cloud
min_bound = point_cloud.get_min_bound()
max_bound = point_cloud.get_max_bound()

print(f"Original bounds:\nMin bound: {min_bound}\nMax bound: {max_bound}")

# Calculate the center of the point cloud
center = (min_bound + max_bound) / 2

print(f"Center of the point cloud: {center}")

# Define the radius for cropping around the center (example: 50 units)
radius = 100.0

# Specify the desired bounds for cropping around the center
min_bound_center = center - radius
max_bound_center = center + radius

print(f"Cropping to bounds:\nMin bound: {min_bound_center}\nMax bound: {max_bound_center}")

# Crop the point cloud
cropped_point_cloud = crop_point_cloud(point_cloud, min_bound_center, max_bound_center)
o3d.visualization.draw_geometries([cropped_point_cloud])
# Save the cropped point cloud to a new .ply file
#o3d.io.write_point_cloud(output_file, cropped_point_cloud)

print(f"Cropped point cloud saved to {output_file}")


import numpy as np
import open3d as o3d
import cv2

def color_overlapping_points(point_cloud_file, image_file, projection_matrix, output_file):
    # Load existing colored point cloud
    point_cloud = o3d.io.read_point_cloud(point_cloud_file)
    lidar_points = np.asarray(point_cloud.points)
    existing_colors = np.asarray(point_cloud.colors)

    # Transform points to image coordinates using the new projection matrix
    transposed_points = lidar_points.T
    xyz = np.vstack((transposed_points, np.ones((1, transposed_points.shape[1]))))

    points = np.matmul(projection_matrix, xyz)

    # Convert to image coordinates
    points = np.array([points[0, :] / points[2, :],
                       points[1, :] / points[2, :]]).T

    # Load new image
    image = cv2.imread(image_file)

    # Extract new colors from the image for overlapping points
    rows, cols, _ = image.shape
    for i, point in enumerate(points):
        px_c, px_r = int(point[0]), int(point[1])
        if 0 <= px_c < cols and 0 <= px_r < rows:
            color = image[px_r, px_c]  # BGR color from the image
            existing_colors[i] = color / 255.0  # Update color for overlapping points

    # Update the point cloud with the new colors
    point_cloud.colors = o3d.utility.Vector3dVector(existing_colors)

    # Visualize the updated point cloud
    o3d.visualization.draw_geometries([point_cloud])

    # Save the updated point cloud as a .ply file
    #o3d.io.write_point_cloud(output_file, point_cloud)

# Example usage:
existing_point_cloud_file = "pakka_vehicle_registered_colour.ply"
#new_image_file = "/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/images/s110_camera_basler_south1_8mm/1688625741_172984730_s110_camera_basler_south1_8mm.jpg"
new_image_file = "/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/images/s110_camera_basler_south2_8mm/1688625741_148698817_s110_camera_basler_south2_8mm.jpg"
new_projection_matrix = np.asarray(
    [[1279.275240545117, -862.9254609474538, -443.6558546306608, -16164.33175985643],
     [-57.00793327192514, -67.92432779187584, -1461.785310749125, -806.9258947569469],
     [0.7901272773742676, 0.3428181111812592, -0.508108913898468, 3.678680419921875]], dtype=np.float32)
p_2 = np.asarray([[1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
                  [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
                  [0.73326062, 0.59708904, -0.32528854, -1.30114325]], dtype=np.float32)
output_file = "vehicle_updated_coloured_point_cloud.ply"

color_overlapping_points(existing_point_cloud_file, new_image_file, p_2, output_file)


import numpy as np
import open3d as o3d
import cv2

def visualize_point_cloud_with_colors(point_cloud, image):
    # Load LiDAR points
    point_cloud = o3d.io.read_point_cloud(point_cloud)
    lidar_points = np.asarray(point_cloud.points)

    transposed_points = lidar_points.T
    # print(points[:3])
    xyz = np.vstack((transposed_points, np.ones((1, transposed_points.shape[1]))))
    # Apply reflection if necessary
    # reflection = points[3, :]
    # points[3, :] = 1
    p_2 = np.asarray([[1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
                                         [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
                                         [0.73326062, 0.59708904, -0.32528854, -1.30114325]], dtype=np.float32)
    p_1 = np.asarray(
            [[1279.275240545117, -862.9254609474538, -443.6558546306608, -16164.33175985643],
             [-57.00793327192514, -67.92432779187584, -1461.785310749125, -806.9258947569469],
             [0.7901272773742676, 0.3428181111812592, -0.508108913898468, 3.678680419921875]], dtype=np.float32)
    p= np.asarray([[1019.929965441548, -2613.286262078907, 184.6794570200418, 370.7180273597151],
                                         [589.8963703919744, -24.09642935106967, -2623.908527352794,
                                          -139.3143336725661],
                                         [0.9841844439506531, 0.1303769648075104, 0.1199281811714172,
                                          -0.1664766669273376]], dtype=np.float32)

    transformation_matrix_vehicle_lidar_to_infra_lidar = np.array([[0.0898674, -0.9959314, -0.0066743, 28.6824274], [0.9958863, 0.0899373, -0.0110319, -54.9192195],
                                                                    [0.0115873, -0.0056554, 0.9999169, -5.373876], [0, 0, 0, 1]], dtype=float)
    vehicle_cam_to_lidar = np.asarray([[0.12672871, 0.12377692, 0.9841849, 0.14573078],  # TBD
                                       [-0.9912245, -0.02180046, 0.13037732, 0.19717109],
                                       [0.03759337, -0.99207014, 0.11992808, -0.02214238],
                                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    vehicle_lidar_to_cam = np.array([[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.23],
              [0.0, 0.0, 1.0, -0.01],
              [0.0, 0.0, 0.0, 1.0]])

    extrinsic = np.matmul(np.linalg.inv(vehicle_cam_to_lidar),  np.linalg.inv(transformation_matrix_vehicle_lidar_to_infra_lidar))
    #perfect working  extrinsic = np.matmul(vehicle_cam_to_lidar,  np.linalg.inv(transformation_matrix_vehicle_lidar_to_infra_lidar))


    vehiclecamintrinsics = np.asarray([[2726.55, 0.0, 685.235, 0.0],
                                       [0.0, 2676.64, 262.745, 0.0],
                                       [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    shata_intr_south = np.asarray([[1029.2795655594014, 0.0, 982.0311857478633, 0.0],
                [0.0, 1122.2781391971948, 1129.1480997238505, 0.0],
                [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    shata = vehicle_cam_to_lidar @ transformation_matrix_vehicle_lidar_to_infra_lidar
    south22infralidar = np.asarray([[0.6353517, -0.24219051, 0.7332613, -0.03734626],
                                    [-0.7720766, -0.217673, 0.5970893, 2.5209506],
                                    [0.01500183, -0.9454958, -0.32528937, 0.543223],
                                    [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    #vehiclecam2infrastructurecam_extrinsics = base @ south22infralidar @ transformation_matrix_vehicle_lidar_to_infra_lidar @ np.linalg.inv(vehicle_cam_to_lidar)
    proj_matrix = np.matmul(vehiclecamintrinsics, extrinsic)
    print(proj_matrix)

    points = np.matmul(proj_matrix, xyz)

    # Convert to image coordinates
    points = np.array([points[0, :] / points[2, :],
                       points[1, :] / points[2, :],
                       ]).T

    # Load image
    image = cv2.imread(image)

    # Extract colors from image
    rows, cols, _ = image.shape
    colors = []
    for point in points:
        px_c, px_r = int(point[0]), int(point[1])
        if 0 <= px_c < cols and 0 <= px_r < rows:
            color = image[px_r, px_c]  # BGR color from the image
            colors.append(color)
        else:
            colors.append([0, 0, 0])  # Set default color if point is outside image boundaries
    colors = np.array(colors) / 255.0  # Normalize colors to [0, 1] range

    # Create Open3D point cloud with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    #o3d.io.write_point_cloud(output_file, pcd)

# Example usage:
#point_cloud_file = "/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/1688625741_146525143_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.pcd"
#point_cloud_file ="/home/uchihadj/CVPR_2025/4DGaussians/data/distendgled_scene/meta_data/1688626890_140238582_s110_lidar_ouster_south.pcd"
point_cloud_file =  "/home/uchihadj/CVPR_2025/4DGaussians/data/distendgled_scene/meta_data/1688626890_140238582_s110_lidar_ouster_south.pcd"
#image_file = "/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/images/s110_camera_basler_south1_8mm/1688625741_172984730_s110_camera_basler_south1_8mm.jpg"
image_file = "/home/uchihadj/CVPR_2025/4DGaussians/data/distendgled_scene/cam03/1688626890_152253885_vehicle_camera_basler_16mm.jpg"
#image_file = "/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/images/s110_camera_basler_south2_8mm/1688625741_148698817_s110_camera_basler_south2_8mm.jpg"
output_file = "south_2_vehicle_scene_non_overlap_sientangled.ply"
visualize_point_cloud_with_colors(point_cloud_file, image_file)

"""
import numpy as np
import open3d as o3d
import cv2

def update_colors_of_point_cloud(point_cloud_file, image_file, projection_matrix, output_file):
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

    # Extract new colors from the image
    rows, cols, _ = image.shape
    new_colors = np.copy(existing_colors)
    for i, point in enumerate(points):
        px_c, px_r = int(point[0]), int(point[1])
        if 0 <= px_c < cols and 0 <= px_r < rows:
            color = image[px_r, px_c]  # BGR color from the image
            new_colors[i] = color / 255.0  # Normalize color to [0, 1] range

    # Update the point cloud with the new colors
    point_cloud.colors = o3d.utility.Vector3dVector(new_colors)

    # Visualize the updated point cloud
    o3d.visualization.draw_geometries([point_cloud])

    # Save the updated point cloud as a .ply file
    #o3d.io.write_point_cloud(output_file, point_cloud)

# Example usage:
#existing_point_cloud_file = "pakka_vehicle_registered_colour.ply"
existing_point_cloud_file = "south_2_vehicle_updated_colours.ply"
new_image_file = "/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/images/s110_camera_basler_south1_8mm/1688625741_172984730_s110_camera_basler_south1_8mm.jpg"
#new_image_file = "/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/images/s110_camera_basler_south2_8mm/1688625741_148698817_s110_camera_basler_south2_8mm.jpg"
new_projection_matrix = np.asarray(
    [[1279.275240545117, -862.9254609474538, -443.6558546306608, -16164.33175985643],
     [-57.00793327192514, -67.92432779187584, -1461.785310749125, -806.9258947569469],
     [0.7901272773742676, 0.3428181111812592, -0.508108913898468, 3.678680419921875]], dtype=np.float32)
p_2 = np.asarray([[1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
                  [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
                  [0.73326062, 0.59708904, -0.32528854, -1.30114325]], dtype=np.float32)
#output_file = "south_2_vehicle_updated_colours.ply"
output_file = "Only_USE_THIS_INIT_south_1_south_2_vehicle_updated_colours.ply"

update_colors_of_point_cloud(existing_point_cloud_file, new_image_file, new_projection_matrix, output_file)
"""
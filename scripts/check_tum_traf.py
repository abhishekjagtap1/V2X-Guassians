import cv2
import numpy as np
# import pcl
import matplotlib.pyplot as plt
#from pypcd import PointCloud
import cv2

# image = plt.imread("/home/uchihadj/4DGaussians/data/images/0000.png")
#image = plt.imread("/home/uchihadj/4DGaussians/data/south_1_2_and_vehicle/images/1688627148_953223175_s110_camera_basler_south1_8mm.png")
#image = plt.imread("/home/uchihadj/TUMtraf/TUMTraf-NeRF/a9_v2x_nerf/train/scene-7/images/0080.png")
#image = cv2.imread("/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/images/s110_camera_basler_south2_8mm/1688625741_148698817_s110_camera_basler_south2_8mm.jpg")
#image = cv2.imread("/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/images/s110_camera_basler_south1_8mm/1688625741_172984730_s110_camera_basler_south1_8mm.jpg")
#image = cv2.imread("/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/images/vehicle_camera_basler_16mm/1688625741_152104872_vehicle_camera_basler_16mm.jpg")
image = cv2.imread("/home/uchihadj/CVPR_2025/4DGaussians/data/distendgled_scene/cam03/1688626890_152253885_vehicle_camera_basler_16mm.jpg")


def points_on_image(undistorted_img, velodyne=False):
    color_scale = 255 / 3

    import open3d as o3d
    #pcd_loader = o3d.io.read_point_cloud("/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/point_clouds/s110_lidar_ouster_south/1688625741_146525143_s110_lidar_ouster_south.pcd")
    #pcd_loader = o3d.io.read_point_cloud("/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/point_clouds/vehicle_lidar_robosense/1688625741_192682743_vehicle_lidar_robosense.pcd")
    #pcd_loader = o3d.io.read_point_cloud("/home/uchihadj/Downloads/tumtraf_v2x_cooperative_perception_dataset/train/point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/1688625741_146525143_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.pcd")
    pcd_loader = o3d.io.read_point_cloud("/home/uchihadj/CVPR_2025/4DGaussians/data/distendgled_scene/meta_data/1688626890_140238582_s110_lidar_ouster_south.pcd")
    # np.save("/home/uchihadj/4DGaussians/data/points_scenes7.npy", points)
    points = np.asarray(pcd_loader.points)
    total_points = np.asarray(pcd_loader.points)

    # points[:3]
    # points = np.array(point_cloud)

    # Transpose for easier manipulation
    transposed_points = points.T
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

    south22infralidar = np.asarray([[0.6353517, -0.24219051, 0.7332613, -0.03734626],
                                        [-0.7720766, -0.217673, 0.5970893, 2.5209506],
                                        [0.01500183, -0.9454958, -0.32528937, 0.543223],
                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    transformation_matrix_vehicle_lidar_to_infra_lidar = np.array([[0.0898674, -0.9959314, -0.0066743, 28.6824274], [0.9958863, 0.0899373, -0.0110319, -54.9192195],
                                                                    [0.0115873, -0.0056554, 0.9999169, -5.373876], [0, 0, 0, 1]], dtype=float)
    vehicle_cam_to_lidar = np.asarray([[0.12672871, 0.12377692, 0.9841849, 0.14573078],  # TBD
                                       [-0.9912245, -0.02180046, 0.13037732, 0.19717109],
                                       [0.03759337, -0.99207014, 0.11992808, -0.02214238],
                                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    extrinsic = np.matmul(np.linalg.inv(vehicle_cam_to_lidar), np.linalg.inv(transformation_matrix_vehicle_lidar_to_infra_lidar))


    vehiclecamintrinsics = np.asarray([[2726.55, 0.0, 685.235, 0.0],
                                       [0.0, 2676.64, 262.745, 0.0],
                                       [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    proj_matrix = np.matmul(vehiclecamintrinsics, extrinsic)
    #repr(proj_matrix)
    print("proj_matrix", repr(proj_matrix))

    # Apply transformation matrix
    points = np.matmul(proj_matrix, xyz)

    # Convert to image coordinates
    points = np.array([points[0, :] / points[2, :],
                       points[1, :] / points[2, :],
                       ]).T
    # points = points.astype(int)

    # Apply transformation matrix to LiDAR points if needed
    lidar_points_transformed = np.matmul(p_2, xyz).T  # Shape (N, 3)

    # Convert LiDAR points to image coordinates
    image_coordinates = lidar_points_transformed[:, :2].astype(int)  # Shape (N, 2)

    # Get image shape
    rows, cols, _ = image.shape

    # Initialize array to store LiDAR points with colors
    lidar_points_with_colors = []

    # Iterate over LiDAR points and retrieve colors from image
    for i, (px_c, px_r) in enumerate(image_coordinates):
        if 0 <= px_c < cols and 0 <= px_r < rows:
            # Retrieve color from image
            color = image[px_r, px_c]

            # Append LiDAR point with color to the list
            lidar_points_with_colors.append(np.hstack((lidar_points_transformed[i], color)))

    # Convert the list of LiDAR points with colors to a NumPy array
    lidar_points_with_colors = np.array(lidar_points_with_colors)
    lidar_points = lidar_points_with_colors[:, :3]
    colors = lidar_points_with_colors[:, 3:] / 255.0  # Normalize colors to range [0, 1]

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(total_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    (rows, cols, channels) = undistorted_img.shape
    for cor in points:
        px_c, px_r = cor

        if cols > px_c and px_c > 0 and rows > px_r and px_r > 0:
            cv2.circle(undistorted_img, (int(px_c), int(px_r)), 1, (140, 70, 255), -1)

    cv2.imshow("Lidar points on image", undistorted_img)
    cv2.waitKey(100000000)


# Example usage:
# undistorted_img = cv2.imread("image.jpg")
# point_cloud_file = "point_cloud.pcd"
# rotation_translation = np.array([[1, 0, 0, 0.0],
#                                  [0, 1, 0, 0.0],
#                                  [0, 0, 1, 0.0]])
# intrinsic = np.array([[focal_length_x, 0, principal_point_x],
#                       [0, focal_length_y, principal_point_y],
#                       [0, 0, 1]])
# points_on_image(undistorted_img, point_cloud_file, rotation_translation, intrinsic)
points_on_image(image)
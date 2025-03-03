# Vehicle-to-Everything Gaussians on The Road (V2X-GOTR) Dataset Preparation

## Overview
The **Vehicle-to-Everything Gaussians on the Road (V2X-GOTR) Evaluation benchmark** is built upon the [TUMTraf-V2X](https://innovation-mobility.com/tumtraf-dataset) dataset. It systematically extracts, processes, and organizes key components from the dataset to support high-fidelity dynamic scene reconstruction. In
order to bolster future research in neural scene rendering and off-axis novel view synthesis in the V2X ecosystem, we provide a consistent benchmark for training and evaluation using just 3 Steps

## 1. Data Directory Setup
To organize the dataset properly, follow these steps:

1) Download the TUMTRAF-V2X dataset from https://innovation-mobility.com/en/project-providentia/a9-dataset/ and setup the directory to store the dataset.

```bash
# Create a directory to store raw data
mkdir -p ./data/TUMTRAF/   
```

Alternatively, you can create a symbolic link to an existing data directory:

```bash
ln -s /path/to/your/data ./data/TUMTRAF
```

Ensure that the dataset is placed correctly within the respective directories before proceeding with further pre-processing.

### Selected Scenarios

V2X-GOTR comprises of nine diverse, handpicked scenarios that tackle a wide range of challenges such as occlusion, blind spots etc..
The selected scenarios are as follows:

| Scene Name                      |
|---------------------------------|
| Ego Vehicle Occlusion           |
| Pedestrian Crossing             |
| Sharp U-Turn Maneuver           |
| U-Turn Maneuver                 |
| RSU Occlusion                   |
| Far Distance Pedestrian         |
| Dense VRU Crossing              |
| Dense VRU with RSU Occlusion    |
| Night Scene                     |

### Sensors and Field of View

The V2X-GOTR dataset includes data from **2 road-side unit cameras (Infrastructure)**, **1 camera from the ego vehicle (EGO-CAV)**, and **LiDAR Point clouds** whose fields of view (FOV) collectively cover the intersection. Below is the mapping of sensor and LiDAR names to their simplified identifiers that will be used in training part of the code:

| Sensor Name                                     | Identifier |
|-------------------------------------------------|---------|
| `s110_camera_basler_south2_8mm`                 | `cam1`  |
| `vehicle_camera_basler_16mm`                    | `cam2`  |
| `s110_camera_basler_south1_8mm`                 | `cam3`  |
| `s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered` | `lidar` |

---

## 2. Pre-processing the Dataset

To preprocess all the selected scenarios, run the following command:

```bash
python V2X_GOTR_Dataset/V2X_GOTR_data_preprocessor.py --dataset "path_to_TUMTRAF-V2X_dataset"
```
After preprocessing, all the scenes will be organized in a structured directory as follows:
```
├── data
|   | pedestrian_crossing
│       | cam1
|     		├── frame_00001.jpg
│     		├── frame_00002.jpg
│     		├── ...
│   	| cam2
│     		├── frame_00001.jpg
│     		├── frame_00002.jpg
│     		├── ...
|       | cam3
│     		├── frame_00001.jpg
│     		├── frame_00002.jpg
│     		├── ...
|       | lidar
│     		├── lidar_timestamp_1.pcd
│     		├── lidar_timestamp_2.pcd
│     		├── ...
|       | metadata
│     		├── metadata_timestamp_1.json
│     		├── metadata_timestamp_2.json
│     		├── ...
│   | New Scene ...
```
## 3. Enabling V2X-GS Training on specific scenes

Inorder to train V2X-Gaussians on specific scenes (for instance on pedestrian crossing), follow the below instructions to enable a robust lidar initialization
```shell
python V2X_GOTR_Dataset/lidar_initialization_per_scene.py \
    --lidar_input_path "/path/to/lidar/frames" \
    --combined_output_path "/path/to/save/combined.pcd" \
    --voxel_size 0.05 \
    --downsampled_output_path "/data/pedestrian_crossing/lidar_downsampled.ply"
```

Important Notes: 
1. It is recomended to keep the ```voxel size``` to ```0.05```, inorder to reproduce the results in the paper. 
2. It is recomended to save the lidar initialization ```downsampled_output_path``` in ```data/example_scene``` directory.

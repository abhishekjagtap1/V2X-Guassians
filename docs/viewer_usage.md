# V2X Gaussians - SIBR Viewer
This doc is heavily borrowed from [3D-GS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip), thanks for the author!


# Windows Support For Viewer

Follow the official instructions in [SIBR Viewer](./submodules/SIBR_viewers/README.md).

## Ubuntu 22.04
For ubuntu follow the below commands to setup SIBR framework for Image-based Rendering applications:

```bash
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
# Project setup
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # add -G Ninja to build faster
cmake --build build -j24 --target install
```
If you train the V2X-Gaussians locally, use the following command to visualize training:
```python
./viewers/bin/SIBR_remoteGaussian_app --port 6018 # port should be same with your trainging code.

```
Visualizing Trained Model 
```
./install/bin/SIBR_gaussianViewer_app -m "path_to_trained_model"
```



# Demo - 1: V2X Agents Collaborative View

![View](/assets/Collaborative_view.gif) 

# Demo - 2: Dynamic Scene Visualization of Infrastructure and Ego-CAV 

Note: Camera keys for SIBR Viewer are available at ```.submodules/SIBR_viewers/camera_pos```, allowing seamless switching between agents' perspectives.


![](/assets/novel_view.gif "") 

![](/assets/Dynamic_video.gif "")



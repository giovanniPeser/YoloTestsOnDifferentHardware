# myYoloTest
Test campaign for Yolov5 and Yolov8 fir different hardware including Nvidia Jetson Nano, Raspberry Pi 4 and Dell XPS 15 7590.

# YOLOV8 on jetson Nano 
Nvidia Jetson Nano is distributed with jetpack 4.6.1, in particular an image containing already the jetpack 4.6.1 can be found, as described https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write. Even if another image is used, unlckily the kernel fot jetson Nano, which is a customized kernel out of mainline is stuck at 4.9.337-tegra version, and essential custom driver like those one for the GPU Tegra X1 are not supported with higher version. This distribution is designed to support python 3.6.9, but this is a problem since ultralytics repository requires python 3.8.
Following the  video at  https://www.youtube.com/watch?v=pAEkHsNkul0 , the following instruction were executed on the target (on Nvidia Jetson Nano flashed with previously cited image):

```
sudo apt update
```

```
sudo apt install build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev libffi-dev libc6-dev
```
```
wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tar.xz
```
Extract the downloaded archive by running the following command:
```
tar -xf Python-3.8.12.tar.xz
cd Python-3.8.12
```
Configure the build process:
```
./configure --enable-optimizations
```
Build Python:
```
make -j4
```
Once the compilation is complete, you can install Python by running the following command:
```
sudo make altinstall
python3.8 --version
```
To avoid to break image dependency with python3.6, create a virtual environment with
```
/usr/local/bin/python3.8 -m venv /home/giovanni/Python-3.8.12/myenv
```
to activate the environment
```
source /home/giovanni/Python-3.8.12/myenv/bin/activate
```
Unluckily to install pytorch, it must be compiled on target, since nvidia does not provide a precompiled package for the target.
As first thing, it must be checked that  cuda is install with:
```
nvcc --version
```
Probably the command give an error since it is installed in the old environment, but not in the one just created.
Since the image provide cuda toolkit, then it is enough to add it to the environment with:
```
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
```
Or it is possible to add the previous line to the activate file (~/Python-3.8.12/myenv/bin/activate) such that the environment will always start exporting those paths.
Now it is possible to install ultralytics
```
pip install ultralytics
```
This action will install also a generic pytorch version for an aarch64, which does not support Nvidia Tegra X1. For this reasonthe following steps shall be followed.
- Uninstall torch 
```
pip unistall torch
```
Then it is the necessary to build the package on the target. In this repo the package "torch-1.10.0-cp38-cp38m-linux_arch64_and_cuda_for_nvidia_tegra_for_Nvidia_Jetson_nano.whl" is provided to allow to avoid the compilation of the target.
Once download, you can simply install it with:
```
pip install torch-1.10.0-cp38-cp38m-linux_arch64_and_cuda_for_nvidia_tegra_for_Nvidia_Jetson_nano.whl
```
Instead, if torch is wanted to be compiled, the following steps shall be followed:
Download pytorch, with a version compatible with the jetpack
```
git clone --branch v1.10.0 --recursive https://github.com/pytorch/pytorch
```
And compile
```
cd torch
sudo env MAX_JOBS=1 ~/Python-3.8.12/myenv/bin/python setup.py install
```
Note that MAX_JOBS=1 must be given since otherwise the compilation fails due to a problem with the RAM
TO make it a bit faster, first compile without myenv (such that parallel process is used until the error) and then run again with MAX_JOBS=1 (It take a long time, more than entire day)
If you want to create your wheel, then you can do:
```
sudo ~/Python-3.8.12/myenv/bin/python setup.py bdist_wheel
```
which create the wheel package in the dist folder.

Then also torchvision shall be uninstalled and recompiled
```
pip unistall torchvision
```
And then download a compatible version of torchvision
```
git clone --branch v0.11.1 https://github.com/pytorch/vision.git
```
and install it 
```
cd torchvision
sudo /home/giovanni/Python-3.8.12/myenv/bin/python setup.py install
```

After the installation, also the yolov5 was downloaded to use it for yolov5 tests. In particular executing the following steps, as described https://docs.ultralytics.com/yolov5/tutorials/running_on_jetson_nano/#deepstream-configuration-for-yolov5 .
```
git clone --branch  915bbf29 https://github.com/ultralytics/yolov5
```
Change requirement to avoid to install other version of pytorch
```
cd yolov5
vi requirements.txt
```
in Parituclar edit the following lines.
```
# torch>=1.8.0
# torchvision>=0.9.0
```
Then install dependencies:
```
sudo apt install -y libfreetype6-dev
pip install -r requirements.txt
```
After that it is necessary to install tensorflow lite and tensorflow.
Tensorflow can be downloaded from ufficial repo here https://developer.download.nvidia.com/compute/redist/jp/v50/tensorflow/ (or you can use the one in this repo)
First install dependencies, as described https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770 .
```
sudo apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran  
pip install --verbose 'protobuf<4' 'Cython<3'
```
Then install the tensorflow package with:
```
pip install tensorflow-2.8.0+nv22.3-cp38-cp38-linux_aarch64.whl
```
Finally install tensorflow-lite with:
```
pip install tflite-runtime
```

After all these steps, we are ready to evaluate performance of YOLOV5 and YOLOV8 on Nidia Jetson Nano.

# Evaluation of Time performances
Once prepared the environment, installing ultralytics, yolov5 requirements and cuda pack if the device has a GPU supporting cuda, then it is possible to evaluate the performance on the different target.
Copy the file evaluate_nn.py in the yolov5 folder:
```
~/myYoloTest/performance_evaluation/evaluate_nn.py ~/yolov5/
```
move in the folder.
```
cd ~/yolov5/
```
Then the dataset to evaluate the time performance shall be downloaded.
For COCO dataset, download the val2017 with
```
wget http://images.cocodataset.org/zips/val2017.zip
```
and unzip it, with:
```
unzip val2017.zip
```

Run it 
```
cd ~/yolov5/
python  ~/yolov5/evaluate_nn.py
```
It is possible to evaluate different target by simply change the variables:
```
is_int8_tflite = False # which define if tflite-mode is wated to be used
mydevice = 'cpu' # which define the type of processing unit 
                 # (if gpu is wanted then it shall be 'cuda' )
model_type = 's' # which defines the type of yolo model
                 # we used only s and n
is_u = True      # which defines if the YOLOv5 model is the ultralytics one
size=640         # which defines the input image size
                 # we used 640 (for 640x640) and 320 for 320x320
yolovers = "5"   # which defines the version of yolo model  used
                 # we used only 5 and 8
```
TO DO: Take the parameters from the commandline and use directly them.

# Evaluation of MAP50 and MAP50-95 performances
For the evaluation of map50-95 two different script were used to evaluate the performances.
evaluate_map50-95.py allows to evaluate the performance of YOLOv8 and YOLOv5U. As the previous script, the network is selected through the same parameters.
Firstly, copy the file evaluate_map50-95.py in the yolov5 folder:
```
cp ~/myYoloTest/performance_evaluation/evaluate_map50-95.py ~/yolov5/
```
Then run the script with:
```
python  ~/yolov5/evaluate_map50-95.py
```
on the standard output the map50-95 and map50 evaluated on coco_test dataset will be printed, for example the following output can be produced:
```
model 8s with image size 320 map50-95 result : 0.588212305925844
model 8s with image size 320 map50 result : 0.759425030874107
```
Instead, to evaluate yolov5 the following command shall be executed from yolov5 folder:
```
python val.py --weights ../yolov5n.pt --data coco.yaml --img 320
```
To have a faster visualization of the results, it is possible to modify the cal.py code, in particular changing the function run, and introducing the print of the result.
In particular by adding the following lines:
```
print(" map50 result :" + str(map50))
print(" map50-95 result :" + str(map))
```
before the return statement of the run function.

Note: the coco_dataset will be downloaded launching the script.
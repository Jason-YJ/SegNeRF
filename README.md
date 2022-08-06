# SegNeRF

## Installation
- Pull SegNeRF repo.
  ```
  git clone --recursive git@github.com:weiyithu/NerfingMVS.git
  ```
  
- Install Mseg 
  
  cd to root path first. Then
  
  ```
  ipython
  !git clone https://github.com/mseg-dataset/mseg-api.git
  !cd mseg-api && sed -i '12s/.*/MSEG_DST_DIR="\/dummy\/path"/' mseg/utils/dataset_config.py
  !cd mseg-api && pip install -e .
  !cd mseg-api && wget --no-check-certificate -O "mseg-3m.pth" "https://github.com/mseg-dataset/mseg-semantic/releases/download/v0.1/mseg-3m-1080p.pth"
  !cd mseg-api && git clone https://github.com/mseg-dataset/mseg-semantic.git
  !cd mseg-api/mseg-semantic && pip install -r requirements.txt
  !cd mseg-api/mseg-semantic && pip install -e .
  ```
  
- Install python packages with anaconda.
  
  ```
  conda create -n SegNeRF python=3.7
  conda activate SegNeRF
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
  pip install -r requirements.txt
  ```
  
- We use COLMAP to calculate poses and sparse depths. However, original COLMAP does not have fusion mask for each view. Thus, we add masks to COLMAP and denote it as a submodule. Please follow https://colmap.github.io/install.html to install COLMAP in `./colmap` folder (Note that do not cover colmap folder with the original version). 

## Usage
- Download 8 ScanNet scene data used in the paper [here](https://drive.google.com/file/d/1eY85_adVY-Z8y4XUG8mi31TJzAfdx59M/view?usp=sharing) and put them under `./data` folder. 
- Run SegNeRF
  ```
  sh run.sh $scene_name
  ```
  The whole procedure takes about 3.5 hours on one NVIDIA GeForce RTX 2080 GPU, including COLMAP, depth priors training, NeRF training, filtering and evaluation. COLMAP can be accelerated with multiple GPUs.You will get per-view depth maps in `./logs/$scene_name/filter`. Note that these depth maps have been aligned with COLMAP poses. COLMAP results will be saved in `./data/$scene_name` while others will be preserved in `./logs/$scene_name`

## Run on Your Own Data!
- Place your data with the following structure:
  ```
  NerfingMVS
  |───data
  |    |──────$scene_name
  |    |   |   train.txt
  |    |   |──────images
  |    |   |    |    001.jpg
  |    |   |    |    002.jpg
  |    |   |    |    ...
  |───configs
  |    $scene_name.txt
  |     ...
  ```
  `train.txt` contains names of all the images. Images can be renamed arbitrarily and '001.jpg' is just an example. You also need to imitate ScanNet scenes to create a config file in `./configs`. Note that `factor` parameter controls the resolution of output depth maps. You also should adjust `depth_N_iters, depth_H, depth_W` in `options.py` accordingly. 

## Acknowledgement
Our code is based on the pytorch implementation of NeRF: [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch). We also refer to [mannequin challenge](https://github.com/google/mannequinchallenge). 




# SiamVFE & SiamGCN-GCA
This is the official implementation of SiamVFE and SiamGCN-GCA for the task of point cloud change detection for city scenes.

## Install
1. Create an anaconda environment with python 3.8.5
`conda create -n "siam" python=3.8.5`

2.  Activate the anaconda environment
`conda activate siam`

3. Install the pytorch version, given your CUDA setup. 
i.e. for pytorch 1.10.1 and CUDA 11.3
`conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge` 

4. Install pytorch-geometric
`conda install -c pyg pyg`

5. Install laspy
`pip install laspy[lazrs,laszip]`

6. Install imbalanced-learn
`conda install -c conda-forge imbalanced-learn`

7. Install matplotlib
`conda install -c conda-forge matplotlib`


## Supported Datasets
- [SHREC2023](https://yhldrf.github.io/pointcloudchange/)

#### SHREC2023
Place the SHREC2023 dataset folder "PCLChange", in the main folder. The structure should be like:

- SiamVFE/PCLChange/
 	- lidar/
 		- 2016/
 		- 2020/
 		- labeled_point_lists_train/
	- synthetic_city_scenes
		- time_a/
		- time_b/
		- labeled_point_lists_train_syn/
 		

## Training
### SiamVFE
- on lidar data
 `python main_siamvfe.py --batch_size 32 --num_workers 4 --data lidar`
 
- on synthetic data
 `python main_siamvfe.py --batch_size 32 --num_workers 4 --data synthetic`
 
### SiamGCN-GCA 
- on lidar data
 `python main_siamgcn_gca.py --batch_size 16 --num_workers 4 --data lidar`
 
- on synthetic data
 `python main_siamgcn_gca.py --batch_size 16 --num_workers 4 --data synthetic`

If out of GPU memory, reduce the batch size.

## Pre-processing
When executing either training script for the first time, the pre-processing step will take place. This is performed only once, to extract the scenes of interest and construct the dataset.

## Evaluate models
To evaluate a model run the training script with the --test_model flag set to True.

Also, select the type of data, lidar or synthetic

 i.e. 
`python main_siamgcn_gca.py --batch_size 4 --num_workers 4 --test_model True --data lidar`

## Acknowledgements
This work relies upon code from  [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and  [SiamGCN](https://github.com/kutao207/SiamGCN).

   
                



# DeepShadow Shadow Extraction Model
This repository is the code 
implementation for ECCV 2022 paper **supplementary**:
"DeepShadow: Neural Shape from Shadow".

The overview of our shadow and light extraction 
architecture is shown below:

<img src="figures/shadow_transformer.png"  style="background-color: white">

## Requirements

* torch > 1.8
* opencv-python > 4.1
* numpy
* kornia > 0.6
* matplotlib
* einops > 0.3.1
* python

Our code was tested using Python 3.7/3.8 under Ubuntu 18.04, with GPU and/or CPU.

## Dataset Used for Training
_skip this if you don't want to train the model_

**Download** [Blobby and Sculptures dataset](https://drive.google.com/drive/folders/1HCW9YDfsFoPxda3GDTjj3L5yuQUXw8a3?usp=sharing)
by Chen et al. (taken from here - https://github.com/guanyingc/SDPS-Net)
_Torch data loading code also taken from here._

**Download** our PhotometricStereo Shadow data - (coming soon!)


## Run Inference using the model
1. Clone the repo -
```bash
git clone https://github.com/asafkar/ps_shadow_extract.git
cd ps_shadow_extract/
```

2. Download the model checkpoint
```bash
# get the checkpoint from the git lfs
git lfs install
git lfs fetch
```

3. Install requirements
```bash
pip install -r requirements.txt
```

4. Use the pretrained model to estimate shadows and lights directions

_refer to run_model_example.ipynb_ 

## Train the model from scratch
1. Download and unzip the data, place all 3 datasets in the same folder.
Indicate the folder when training by using arg _--base_dir_

3. Train the model
```bash
CUDA_VISIBLE_DEVICES=<gpus> python -m torch.distributed.run --nproc_per_node=<num_gpus> train.py --base_dir=<dir>

```



## Citation
If you use the model or dataset in your own research, please cite:
```
@inproceedings{karnieli2022deepshadow,	
		title={DeepShadow: Neural shape from shadows},
		author={Asaf Karnieli, Ohad Fried, Yacov Hel-Or},	
		year={2022},	
		booktitle={ECCV},
}
```




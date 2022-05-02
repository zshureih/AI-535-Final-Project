# AI-535-Final-Project


# Setup Tutorial 


## Setup conda environment
conda create -n "535-final" -y python=3.7
source activate 535-final 


This was compatible with the environment on my local system.
The minimum pytorch version required for using the Transformer model is 1.2.0, but if you want to use more recent versions go ahead and modify accordingly
## install pytorch (cuda recommended)
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch


## install pip and download the data (40GB zip)
pip install gdown 
gdown https://drive.google.com/uc?id=17v86luM3g0Fo7bU2uzV0i0klvrLRyglB -O eval5_dataset_1_of_6.zip

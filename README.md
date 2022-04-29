# AI-535-Final-Project


# Setup Tutorial 


## Setup conda environment
conda create -n "535-final" -y python=3.7
source activate 535-final 


## install pytorch (cuda recommended)
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch


## install pip and download the data (40GB zip)
pip install gdown 
gdown https://drive.google.com/uc?id=17v86luM3g0Fo7bU2uzV0i0klvrLRyglB -O eval5_dataset_1_of_6.zip
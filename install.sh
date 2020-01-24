conda create -n prat python=3.6
conda activate prat
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge opencv
pip install tensorboard
conda install -c conda-forge imgaug
conda install albumentations -c albumentations
conda install -c conda-forge matplotlib

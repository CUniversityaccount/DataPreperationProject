
conda create -n DP python=3.9 -y
conda install -n DP ipykernel --update-deps --force-reinstall -y

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
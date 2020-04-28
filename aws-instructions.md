# Instructions to build a working environment based on the Deep Learning Base AMI (Ubuntu)

Instance must be at least a t2.medium

```bash
# sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7.6
pip3 install pipenv
export PATH=/home/ubuntu/.local/bin:$PATH
/usr/bin/python3 -m pip install --upgrade pip
ssh-keygen -t rsa -b 4096 -C "lindsay.m.edwards@gmail.com"
cat .ssh/id_rsa.pub # copy the key and paste it into github
git clone git@github.com:notreallyme2/tx-metalearning.git
git clone git@github.com:notreallyme2/torch-templates.git
git clone https://github.com/unlearnai/representation_learning_for_transcriptomics.git
cd tx-metalearning
export PIP_NO_CACHE_DIR=false # stops MemoryErrors
pip3 install pytorch-lightning torch torchvision matplotlib jupyter jupyterlab
jupyter-lab --no-browser --port 8888
```
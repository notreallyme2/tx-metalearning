# Instructions to build a working environment based on the Deep Learning Base AMI (Ubuntu)

## To create a GPU instance


```bash
nvidia-smi # check for a working GPU
/usr/bin/python3 -m pip install --upgrade pip
export PATH=/home/ubuntu/.local/bin:$PATH
ssh-keygen -t rsa -b 4096 -C "lindsay.m.edwards@gmail.com"
cat .ssh/id_rsa.pub # copy the key and paste it into github
git clone git@github.com:notreallyme2/tx-metalearning.git
git clone git@github.com:notreallyme2/torch-templates.git
git clone https://github.com/unlearnai/representation_learning_for_transcriptomics.git
cd tx-metalearning
pip3 --no-cache-dir install pytorch-lightning torch torchvision matplotlib jupyter jupyterlab
jupyter-lab --no-browser --port 8888
```

Each time you restart your instance you will need to update the PublicDNS and remount the drives.
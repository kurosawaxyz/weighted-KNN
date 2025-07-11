# Weighted-KNN

## Setup conda environment
```bash
conda create -n wknn python=3.10 -y
conda activate wknn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Run the code
```bash
python bin/train.py --data <data-name> --config <config-name>
```

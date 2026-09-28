# ResEmoteNet: Bridging Accuracy and Loss Reduction in Facial Emotion Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resemotenet-bridging-accuracy-and-loss/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=resemotenet-bridging-accuracy-and-loss)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resemotenet-bridging-accuracy-and-loss/facial-expression-recognition-on-fer2013)](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013?p=resemotenet-bridging-accuracy-and-loss)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resemotenet-bridging-accuracy-and-loss/facial-expression-recognition-on-raf-db)](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db?p=resemotenet-bridging-accuracy-and-loss)

A new network that helps in extracting facial features and predict the emotion labels.

The emotion labels in this project are:
 - Happiness 😀
 - Surprise 😦
 - Anger 😠
 - Sadness ☹️
 - Disgust 🤢
 - Fear 😨
 - Neutral 😐


## Table of Content:

 - [Installation](#installation)
 - [Usage](#usage)
 - [Checkpoints](#checkpoints)
 - [Results](#results)
 - [License](#license)


## Installation

1. Create a Conda environment.
```bash
conda create --n "fer"
conda activate fer
```

2. Install Python v3.10 using Conda.
```bash
conda install python=3.10
```

3. Clone the repository.
```bash
git clone https://github.com/ArnabKumarRoy02/ResEmoteNet.git
```

4. Install the required libraries.
```bash
pip install -r requirements.txt
```

For the NVIDIA RTX 5090 server, use the one-command installer below. RTX 5090
is a Blackwell GPU, so the installer uses the PyTorch 2.7.1 and torchvision
0.22.1 CUDA 12.8 wheels. The old PyTorch 2.1.2/CUDA 12.1 pins in
`requirements.txt` should not be used for this server. The installer creates a
local `.venv`, installs the dependencies, and runs a real CUDA model forward
test:

```bash
cd /mnt/data/yanyi2025/cyj/res_repo
bash setup_server.sh
source .venv/bin/activate
```

If Python is installed under another name, set it when running the script:

```bash
PYTHON_BIN=python3.10 bash setup_server.sh
```

The setup script does not install or modify system packages. If `dlib` fails
to build, install the system build tools once and rerun it:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev
bash setup_server.sh
```

## Usage

Run the file.
```bash
cd train_files
python ResEmoteNet_train.py
```

For the server-side FER2013 directory layout shown in the project setup:

```text
/mnt/data/yanyi2025/cyj/fer2013_img/
    train/<class_name>/...
    val/<class_name>/...
    test/<class_name>/...
```

run the ImageFolder training script from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python train_files/ResEmoteNet_folder_train.py \
    --data-root /mnt/data/yanyi2025/cyj/fer2013_img \
    --gpu 0 \
    --output-dir /mnt/data/yanyi2025/cyj/fer2013/resemotenet
```

The script uses only physical GPU 0, evaluates validation data without random
augmentation, follows the paper's batch size 16, 80 epochs, SGD, initial
learning rate `1e-3`, cross-entropy loss, and factor-0.1 plateau scheduler, and
writes `best_model.pth`, `metrics.csv`, `class_names.json`, and the one-time
held-out `test_metrics.json` to the output directory. The paper does not report
the plateau scheduler's patience; the script defaults to 5 epochs. It expects
exactly seven class folders in each split; `ImageFolder` assigns indices in
sorted folder-name order.

## Checkpoints
All of the checkpoint models for FER2013, RAF-DB and AffectNet-7 can be found [here](https://drive.google.com/drive/folders/1Daxa6d1-XFxxpg6dyxYl4V-anfiHwtqK?usp=sharing).

## Results

 - FER2013:
   - Testing Accuracy: **79.79%** (SoTA - 76.82%)
 - CK+:
   - Testing Accuracy: **100%** (SoTA - 100%)
 - RAF-DB:
   - Testing Accuracy: **94.76%** (SoTA - 92.57%)
 - FERPlus:
   - Testing Accuracy: 91.64% (SoTA - **95.55%**)
 - AffectNet (7 emotions):
   - Testing Accuracy: **72.93%** (SoTA - 69.4%)

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

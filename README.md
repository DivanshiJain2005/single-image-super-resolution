# Edge Priors Guided Deep Unfolding Network for Image Resolution
Our project dedicated to use edge prior as constraint for better guiding image reconstruction process, which could enchance the final reconstruction effects.
## Folder Structure of Project
```
├─main.py

├─option.py

├─requirements.txt

├─README.md

├─trainer.py

├─utility.py


├─loss

│  ├─_init_.py

│  ├─adversarial.py

│  ├─vgg.py


├─models

│  ├─_init_.py

│  ├─denoisingModule.py

│  ├─epgdun.py

│  ├─edgefeatureextractionModule.py

│  ├─edgemap.py

│  ├─intermediatevariableupdateModule.py

│  ├─residualprojectionModule.py

│  ├─textureReconstructionModule.py

│  ├─variableguidereconstructionModule.py


├─mydata

│  ├─_init_.py

│  ├─benchmark.py

│  ├─common.py

│  ├─div2k.py

│  ├─myDataLoader.py

│  ├─srdata.py
```

## Package Dependencies
This project is built with Pytorch 1.0.1, Python3.7,CUDA10.0. For package dependencies, you can install them by:

`pip install -r requirements.txt`

## Content of requirements.txt

```
torch~=1.0.1
matplotlib~=3.0.3
numpy~=1.16.2
scipy~=1.2.1
scikit-image~=0.14.2
imageio~=2.5.0
tqdm~=4.31.1
torchvision~=0.2.0
scikit-learn~=0.20.3
```

## Training
To train base model on DIV2K, we use 3090 GPU and run for 300 epochs:
```

Link to colab: https://colab.research.google.com/drive/1ad1TGHM1mvYwOeYMbk3c7LKSKgLWT3nF?usp=sharing
python main.py --data_train DIV2K --epochs 300 --save_results SAVE_RESULTS
```
If you want to train model with different scales, you can add additional parameter like (default scale is x2):
```
python main.py --data_train DIV2K --epochs 300 --save_results SAVE_RESULTS --scale 4
```




new commands:
train:
python main.py \
  --dir_data data \
  --data_train DIV2K \
  --data_test Set5 Set14 BSD100 Urban100 Manga109 \
  --scale 4 \
  --n_feats 128 \
  --epochs 300 \
  --batch_size 16 \
  --save EPGDUN_x4 \
  --save_models \
  --save_results \
  --rgb_range 255 \
  --patch_size 96 \
  --decay_type Mstep_150_225_275 \
  --lr 1e-4 \
  --ext img \
  --n_threads 4



test:
python main.py \
  --dir_data data \
  --data_test Set5 Set14 BSD100 Urban100 Manga109 \
  --scale 4 \
  --test_only \
  --save_results \
  --rgb_range 255 \
  --resume -1 \
  --load EPGDUN2 \
  --self_ensemble \
  --ext img \
  --n_threads 0

To reliably hit 32+ (x4) and 38+ (x2), you need one of:

Option 1 — Increase model capacity (recommended)

In epgdun.py, increase T and feature width:


self.T = 8  # more iterations
And in option.py, change default features:


parser.add_argument('--n_feats', type=int, default=128, ...)  # was 64
Option 2 — Switch to a proven backbone like EDSR-baseline which is already implemented in many SR repos and reliably hits these targets.

Option 2 — Switch to a proven backbone like EDSR-baseline which is already implemented in many SR repos and reliably hits these targets.

Option 3 — Train longer on Colab with GPU — your MX450 laptop GPU will be very slow for 600 epochs; Colab's T4/V100 would be 10–20× faster.
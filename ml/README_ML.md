# ResNet-18 Training (Graphs)

This folder contains scripts to export EMG plots into image datasets, split them into train/val/test, and train a ResNet-18 classifier.

## 1) Export images from EMG trials

Creates 7 images per trial (one per movement label) for EMG segments and polar plots.
Outputs two folders: `emg/` and `polar/`, each containing class subfolders.

```
python export_resnet_images.py --out dataset_raw --ds 1
```

## 2) Split into train/val/test

```
python split_dataset.py --src dataset_raw/emg --dst dataset_emg --train 0.7 --val 0.15 --seed 42
python split_dataset.py --src dataset_raw/polar --dst dataset_polar --train 0.7 --val 0.15 --seed 42
```

Optional: use symlinks instead of copies (fast, saves space):

```
python split_dataset.py --src dataset_raw/emg --dst dataset_emg --train 0.7 --val 0.15 --seed 42 --link
```

## 3) Install dependencies

CPU only:

```
pip install torch torchvision
```

GPU: use the command from https://pytorch.org/get-started/locally/

## 4) Train ResNet-18

```
python train_resnet18.py --data dataset_emg --epochs 10 --batch 32 --lr 1e-3 --out runs_emg
python train_resnet18.py --data dataset_polar --epochs 10 --batch 32 --lr 1e-3 --out runs_polar
```

Outputs:
- `runs_emg/resnet18_graphs.pt`
- `runs_emg/classes.txt`

## Notes
- Images are saved at 224x224 and normalized with ImageNet stats.
- Edit `train_resnet18.py` if you want different augmentations or a new model.

# Icesfp

PyTorch implementation of Icesfp for surface normal estimation of ice objects from polarization images.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Use a PyTorch version that matches your CUDA environment.

## Dataset

The code expects the dataset root to contain an `icedataset` folder:

```text
data/
`-- icedataset/
    `-- IceApple_restructured/
        |-- I-0/
        |-- I-45/
        |-- I-90/
        |-- I-135/
        |-- I-sum/
        |-- C/
        |-- params/
        |   |-- DoLP/
        |   `-- AoLP/
        |-- synthesis-normals/
        |-- masks/
        `-- normals-png/
```

Each object folder should follow the same structure. The `C/` folder stores the precomputed confidence maps.

## Training

```bash
python train.py -dataset_dir data -code_dir . -batch_size 5
```

Checkpoints are saved to:

```text
CheckPoints/
```

Training logs and visual results are saved to:

```text
runs/
results/
```

## Evaluation

```bash
python evaluation.py -dataset_dir data -code_dir . -batch_size 5 -checkpoint CheckPoints/best-model.pth
```




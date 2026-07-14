# IceSfP: Structure-Aware Consistency Priors for Shape from Polarization in Complex Media

ICML 2026

[![Paper](https://img.shields.io/badge/Paper-ICML%202026-blue)](https://proceedings.mlr.press)
[![Python](https://img.shields.io/badge/Python-3.8+-orange)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)


---
  <img src="./ice_optical.png" width="50%">
</p>


## Overview

Recovering surface normals from single-view polarization images in complex media remains a fundamental challenge in computer vision. This work focuses on ice, a representative complex medium where birefringence, multiple scattering, and anisotropic light transport can corrupt polarization measurements and break the assumptions of classical shape-from-polarization methods.

We propose a structure-aware polarization consistency prior based on the autocorrelation function to capture local spatial coherence in AoLP. We further design a dual-branch network, IceSfP, which integrates raw polarization observations and physics-based normal priors through cross-modal attention and multi-scale feature fusion. We also introduce a real-world ice SfP dataset for benchmarking this challenging setting.

The key idea is that not all polarization signals are equally reliable in complex media. By quantifying where AoLP exhibits strong local spatial autocorrelation, the network can be guided toward structurally coherent regions while suppressing unreliable cues caused by volumetric scattering.

## Contributions

- **Structure-aware polarization consistency prior**: constructed from AoLP autocorrelation and combined with cross-modal reliability attention to selectively weight reliable physics-based normal priors.
- **IceSfP network**: a dual-branch architecture that adaptively fuses raw polarization observations and physics-based priors for robust surface normal estimation in complex media.
- **IceSfP dataset**: a real-world ice SfP dataset with ground-truth surface normals and polarization observations for benchmarking learning-based SfP methods in complex media.

## Method

### Architecture

IceSfP adopts a dual-branch design:

| Component | Description |
| --- | --- |
| Raw polarization branch | Takes polarization observations and extracts multi-scale features with an EPSANet50 backbone and ASPP. |
| Physics prior branch | Encodes candidate normal maps derived from the Fresnel model and concatenates them with the structure-aware consistency prior. |
| CRA module | Performs cross-modal attention at the deepest feature scale to enhance reliable physics-based priors. |
| Multi-scale feature fusion | Integrates features from both branches at multiple resolutions through skip connections. |
| SPADE-enhanced decoder | Preserves local spatial details from polarization images through spatially adaptive normalization. |

### Polarization Consistency Prior

The consistency prior measures the local spatial coherence of the angle of linear polarization (AoLP):

- **Autocorrelation function (ACF)**: uses a DoLP-weighted double-angle vector representation to measure AoLP consistency within local neighborhoods.
- **Stationary wavelet transform (SWT)**: captures multi-scale directional discontinuities and complex texture structures.
- **Consistency map**: combines the normalized correlation decay scale and high-frequency energy to generate a pixel-wise reliability map.

This prior guides the network toward structurally coherent polarization regions and improves robustness in complex media.

## Dataset

Dataset link: https://pan.quark.cn/s/db59b290943f  
Extraction code: `RHP2`

The IceSfP dataset contains:

- High-precision ground-truth surface normals obtained from a structured-light 3D scanner with 0.1 mm accuracy.
- Polarization images captured by a FLIR polarization camera at 2448 x 2048 resolution.
- Real ice samples prepared by slow freezing with silicone molds to preserve realistic scattering effects.

Expected dataset structure:

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


## Installation

```bash
git clone https://github.com/ykmmm0/IceSfP.git
cd IceSfP

conda create -n icesfp python=3.8
conda activate icesfp

pip install -r requirements.txt
```


## Citation

If this work is useful for your research, please cite:

```bibtex
@inproceedings{yu2026icesfp,
  title     = {Structure-Aware Consistency Priors for Shape from Polarization in Complex Media},
  author    = {Yu, Kaimin and Wang, Puyun and He, Huayang and Wu, Xianyu},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  address   = {xxx}
  volume    = {--},
  series    = {Proceedings of Machine Learning Research},
  address   = {Seoul, South Korea},
  publisher = {PMLR}
}
```

## Acknowledgements

This work was supported by Fuzhou University and the Research Institute of Highway, Ministry of Transport.

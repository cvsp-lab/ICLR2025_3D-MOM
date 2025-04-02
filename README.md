<h2 align="center">Optimizing 4D Gaussians for Dynamic Scene Video<br>from Single Landscape Images</h2>
<p align="center">
  <a href="https://www.pnu-cvsp.com/members/inhwan"><strong>In-Hwan Jin</strong></a>
  ·  
  <a href="https://www.pnu-cvsp.com/members/haesoo"><strong>Haesoo Choo</strong></a>
  ·  
  <a href="https://www.pnu-cvsp.com/members/shj"><strong>Seong-Hun Jeong</strong></a>
  ·    
  <strong>Park Heemoon</strong>
  ·  
  <strong>Junghwan Kim</strong>
  ·  
  <strong>Oh-joon Kwon</strong>
  ·  
  <a href="https://www.pnu-cvsp.com/prof"><strong>Kyeongbo Kong</strong></a>
  <br>
  ICLR 2025
</p>

<p align="center">
  <a href="https://cvsp-lab.github.io/ICLR2025_3D-MOM/"><strong><code>Project Page</code></strong></a>
  <a href="https://iclr.cc/virtual/2025/poster/30162"><strong><code>ICLR Paper</code></strong></a>
  <a href="https://github.com/InHwanJin/3DMOM"><strong><code>Source Code</code></strong></a>
</p>
<div align='center'>
  <br><img src="3D-MOM_title.gif" width=70%>
  <br>Generated Dynamic Scene Video from 3D-MOM.
</div>

## Setup

### Environment Setup
Clone the source code of this repo.
```shell
git clone https://github.com/InHwanJin/3DMOM.git
cd 3d-MOM
git submodule update --init --recursive
```

Installation through pip is recommended. First, set up your Python environment:
```shell
conda create -n 3D-MOM python=3.7 
conda activate Gaussians4D
```
Make sure to install CUDA and PyTorch versions that match your CUDA environment. We've tested on NVIDIA GeForce RTX 3090 with PyTorch  version 1.13.1.
Please refer https://pytorch.org/ for further information.

```shell
pip install torch
```

The remaining packages can be installed with:

```shell
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```


### Download Checkpoints
We use the pre-trained Flow estimation and Video generator model. You can download them [here](https://github.com/xingyi-li/3d-cinemagraphy?tab=readme-ov-file) and [here](https://github.com/jeolpyeoni/StyleCineGAN).

After downloading, place the models in the `ckpt` folder inside the respective directories under `thirdparty`.


## Preprocess Datasets


## Training



## 📖 Citation
<!-- If you find this code useful for your research, please consider to cite our paper:) -->

```bibtex
@inproceedings{jinoptimizing,
  title={Optimizing 4D Gaussians for Dynamic Scene Video from Single Landscape Images},
  author={Jin, In-Hwan and Choo, Haesoo and Jeong, Seong-Hun and Heemoon, Park and Kim, Junghwan and Kwon, Oh-joon and Kong, Kyeongbo},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

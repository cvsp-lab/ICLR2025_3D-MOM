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

## Environment setup 
```
git clone https://github.com/InHwanJin/3DMOM.git
cd 3d-MOM
git submodule update --init --recursive
conda create -n 3D-MOM python=3.7 
conda activate Gaussians4D

pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```
In our environment, we use pytorch=2.4.1+cu121

## Usage



## Release

- [ ] Code (around March 2025)

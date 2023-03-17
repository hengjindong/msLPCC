# Scalable Multimodality LiDAR Point Cloud Compression (msLPCC) 

## env list

```
python 3.8.12
h5py 3.6.0
numpy 1.21.2
open3d 0.13.0
plyfile 0.7.4
torch 1.7.1
pytorch3d 0.6.0
spconv 1.2.1
numpngw 0.1.2
```

## Dataset pre-procesing

### Scalable decoupled dataset processing

**Run command:**

```python
python semantic_dataset.py
```

**Prameter set:**
dataset_name: sequences name 00-21
split: train / test
num_points: number of FPS
**Default set:**
get_path() sets 1 frame every 10 frames. (10 frames is faster to learn, 5 frames is slightly better)
fps() sets the fps downsampling to 65536 points. (2048 * 32)

### Depth & Seg Depth processing

**Run command:**

```python
python dataset_process.py
```

**Prameter set:**
dataset_name: sequences name 00-21
split: train / test
depth_width depth_hight: depth width & hight
**Default set:**
depth_width depth_hight 64 64

### 不同LiDAR点云范围

**Run command:**

```python
python semantic_dataset.py
```

**Prameter set:**
dataset_name 更改处理的序列
split 更改处理训练集或者测试集
farthest_sample_pytorch3d 更改fps采样点数量
remove_outof_range 更改点云范围
**Default set:**
range设置 35 25

## msLPCC训练

### 点云模态训练

**Run command:**

```python
python TRAIN_P.py 
```

batch_size 设置训练batch
model 设置训练模型名称 ./models/model_name.py
epoch 设置训练周期
learning_rate 设置训练学习率
latent_size 设置潜在特征值长度
lamb 设置保存名称
dataset_path 设置训练数据集地址
**Default set:**

```python
python TRAIN_P.py --batch_size=32 --model=B_PCT_PCC --epoch=100 --learning_rate=0.0004 --bottleneck_siz=256
```

### 深度图模态训练

**Run command:**

```python
python TRAIN_PC2D.py 
```

**Default set:**

```python
python TRAIN_PC2D.py --batch_size=32 --model=PC2D_EnB --epoch=100 --learning_rate=0.001 --bottleneck_siz=256
```

### 分割图模态训练

**Run command:**

```python
python TRAIN_PC2D.py 
```

**Default set:**

```python
python TRAIN_PC2D.py --batch_size=32 --model=PC2Dsem_EnB --epoch=100 --learning_rate=0.001 --bottleneck_siz=256
```

### 多模态对齐训练

**Run command:**

```python
python TRAIN_PDS_A.py 
```

**Default set:**
depth_enc_model 读取 深度图模态特征编码器
sem_enc_model 读取 深度图模态特征编码器

```python
python TRAIN_A.py --batch_size=32 --epoch=100 --learning_rate=0.001
```

### 多模态融合训练

**Run command:**

```python
python TRAIN_PDS.py 
```

**Default set:**
depth_enc_model 读取 深度图模态特征编码器
sem_enc_model 读取 深度图模态特征编码器

```python
python TRAIN_PDS.py --batch_size=32 --model=PCT_D_S_PCC_SR --epoch=200 --learning_rate=0.0001 --bottleneck_siz=256
```

### 测试

**Run command:**

```python
python TEST_PDS_MS
```

**Default set:**
depth_enc_model 读取 深度图模态特征编码器 PCT_PC2D_EnB
sem_enc_model 读取 深度图模态特征编码器 PCT_PC2Dsem_EnB
model_1 读取 融合后的点云模态和融合模块 PCT_D_S_PCC_SR
list_level 设置可扩展层数

## 对比实验训练&测试

### Yan2019

```python
python TRAIN_P.py --batch_size=32 --model=B_PN_PCC --epoch=100 --learning_rate=0.0004 --bottleneck_siz=256
```

### Huang2019

```python
python TRAIN_P.py --batch_size=32 --model=B_PNPP_PCC --epoch=100 --learning_rate=0.0004 --bottleneck_siz=256
```

### Guo2021

```python
python TRAIN_P.py --batch_size=32 --model=B_NGS_PCC --epoch=100 --learning_rate=0.0004 --bottleneck_siz=256
```

### Liang2022

```python
python TRAIN_P.py --batch_size=32 --model=B_Trans_PCC --epoch=100 --learning_rate=0.0004 --bottleneck_siz=256
```

### Zhang2022

```python
python TRAIN_P.py --batch_size=32 --model=B_PCT_PCC --epoch=100 --learning_rate=0.0004 --bottleneck_siz=256
```

### 测试

**Run command:**

```python
python TEST_P
```

**Default set:**
model_name 读取模型
list_level 设置可扩展层数

## 消融实验训练

### 多模态模块消融实验

**点云模态&深度图模态**
训练Run command:

```python
python TRAIN_PD.py 
```

训练Default set:
depth_enc_model 读取 深度图模态特征编码器

```python
python TRAIN_PD.py --batch_size=32 --model=PCT_PC2D_PCC_SR --epoch=200 --learning_rate=0.0001 --bottleneck_siz=256
```

Test command:

```python
python TEST_PD.py 
```

**点云模态&分割图模态**
Run command:

```python
python TRAIN_PS.py 
```

Default set:
sem_enc_model 读取 深度图模态特征编码器

```python
python TRAIN_PS.py --batch_size=32 --model=PCT_PC2D_PCC_SR --epoch=200 --learning_rate=0.0001 --bottleneck_siz=256
```

Test command:

```python
python TEST_PS.py 
```

**点云模态&深度图模态&分割图模态**
Run command:

```python
python TRAIN_PDS.py 
```

Default set:
depth_enc_model 读取 深度图模态特征编码器
sem_enc_model 读取 深度图模态特征编码器

```python
python TRAIN_PDS.py --batch_size=32 --model=PCT_D_S_PCC_SR --epoch=100 --learning_rate=0.0001 --bottleneck_siz=256
```

Test command:

```python
python TEST_PDS_MS.py 
```

### 可扩展解耦模块消融实验

**Run command:**
ms_method 设置解耦模块方式 FPS / KNN
depth_enc_model 读取 深度图模态特征编码器 PCT_PC2D_EnB
sem_enc_model 读取 深度图模态特征编码器 PCT_PC2Dsem_EnB
model_1 读取 融合后的点云模态和融合模块 PCT_D_S_PCC_SR

```python
python TEST_PDS_MS.py
```

### 编码特征值长度消融实验

**Train command:**
设置不同bottleneck_size

```python
python TRAIN_P.py --batch_size=32 --model=B_PCT_PCC --epoch=100 --learning_rate=0.0004 --bottleneck_siz=256
```

**Test command:**

```python
python TEST_P
```

### LiDAR点云范围消融实验

**Train command:**
dataset中设置不同LiDAR点云范围数据集

```python
python TRAIN_P.py --batch_size=32 --model=B_PCT_PCC --epoch=100 --learning_rate=0.0004 --bottleneck_siz=256
```

**Test command:**

```python
python TEST_P
```

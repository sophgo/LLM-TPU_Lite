# Qwen1.5

## 目录
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 模型编译](#4-模型编译)
  - [5. 模型推理](#5-模型推理)

## 1. 简介
Qwen1.5 是Qwen的第二代版本，它是开源中英双语对话模型，关于它的特性，请前往源repo查看：https://huggingface.co/Qwen。本例程对Qwen进行移植，使之能在SOPHON BM1688或CV186AH上进行推理测试。其中DDR容量必须是大于等于8GB。


## 2. 特性
* 支持BM1688、CV186AH
* 支持INT4模型编译和推理
* 支持基于pybing推理的Python例程

## 3. 运行环境准备

SoC环境上，参考如下命令修改设备内存。
```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/09/11/13/DeviceMemoryModificationKit.tgz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_{vx.x}.tar.xz #vx.x是版本号
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息
./memory_edit.sh -c -npu 6800 -vpu 0 -vpp 40 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/boot.itb /boot/boot.itb && sync
sudo reboot
```

### 3.2 编译环境

TPU-MLIR和docker准备

  ```bash
  docker pull sophgo/tpuc_dev:latest
  # 这里将本级目录映射到docker内的/workspace目录,用户需要根据实际情况将demo的目录映射到docker里面
  # myname只是举个名字的例子, 请指定成自己想要的容器的名字
  docker run --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest
  # 此时已经进入docker，并在/workspace目录下
  pip install tpu_mlir
  ```

## 4. 模型编译

### 下载Qwen1.5官方代码

**注：** Qwen1.5-1.8B官方库50G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-AWQ
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

### 4.2 bmodel编译

```bash
llm_convert.py -m /workspace/Qwen1.5-1.8B-Chat-AWQ -s 512 -q w4bf16 -c bm1688 -o qwen1.5_1.8b
```

编译成功之后，模型将会存放在`qwen1.5_1.8b`目录下。

## 5. 模型推理
```bash
python python_demo/pipeline.py --model_path your_bmodel_path -c token_config
```

## 常见问题

1. 在soc中怎么编译demo?

``` shell
pip3 install pybind11 transformers
# 进入python_demo目录
mkdir build
cd build
cmake ..
make
cp *.so ../
cd ..
python3 chat.py --model_path qwen1.5-1.8b_bm1688_int4_2core.bmodel -c token_config
```

# Qwen2

## 目录
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 模型编译](#4-模型编译)
  - [5. 模型推理](#5-模型推理)

## 1. 简介
Qwen2是Qwen的系列模型，它是开源中英双语对话模型，关于它的特性，请前往源repo查看：https://huggingface.co/Qwen。本例程对Qwen2进行移植，使之能在SOPHON BM1688或CV186AH上进行推理测试。


## 2. 特性
* 支持BM1688、CV186AH
* 支持INT4模型编译和推理
* 支持基于pybing推理的Python例程

## 3. 运行环境准备

### 3.1 运行环境
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
编译模型建议在x86服务器上通过docker进行
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
## 4.1 获取onnx
### 4.1.1 下载Qwen2.5官方代码

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-1.5B-Instruct
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

### 4.1.2 修改官方代码：
本例程的`files`目录下提供了修改好之后的`modeling_qwen2.py`。(transformers请更新到4.41.2以上)可以直接替换掉原仓库的文件：

```bash
pip install transformers_stream_generator einops tiktoken accelerate torch==2.0.1+cpu torchvision==0.15.2 transformers==4.41.2
cp files/Qwen2.5-7B-Instruct/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```

### 4.1.3 导出onnx
- 导出所有onnx模型，如果过程中提示缺少某些组件，直接**pip install**组件即可

```bash
python3 export_onnx.py --model_path your_torch_model --seq_length 512
```
此时有大量onnx模型被导出到本例程中`compile/tmp/onnx`的目录。

### 4.2 bmodel编译

```bash
./compile.sh --name qwen2-1.5b --seq_length 512 --mode int4 --addr_mode io_alone
```


## 5. 模型推理

- 进入python_demo目录
``` shell
pip3 install pybind11
mkdir build && cd build
cmake .. && make
cp *.so ..
cd ..
```

```bash
python python_demo/chat.py --model_path your_bmodel_path
```




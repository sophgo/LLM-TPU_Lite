![](./assets/sophgo_chip.png)

# 介绍

本项目实现算能BM1688和CV168AH部署各类开源`生成式AI模型`，主要是70亿参数量以内的大语言模型。

我们已经部署过的开源模型如下：

|Model                |INT4                |Huggingface Link                                                          |
|:-                   |:-                  |:-                                                                        |
|ChatGLM3-6B          |:white\_check\_mark:|[LINK](https://huggingface.co/THUDM/chatglm3-6b)                          |
|Qwen1.5-1.8B         |:white\_check\_mark:|[LINK](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat)                     |
|Qwen2-1.5B           |:white\_check\_mark:|[LINK](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)                   |
|Qwen2-7B             |:white\_check\_mark:|[LINK](https://huggingface.co/Qwen/Qwen2-7B-Instruct)                     |
|Qwen2.5-1.5B         |:white\_check\_mark:|[LINK](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)                 |
|Llama2-7B            |:white\_check\_mark:|[LINK](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)              |
|MiniCPM-2B           |:white\_check\_mark:|[LINK](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)                |
|Phi-3-3.8B           |:white\_check\_mark:|[LINK](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)           |
|Gemma2-2B            |:white\_check\_mark:|[LINK](https://huggingface.co/google/gemma-2-2b-it)                       |
|OpenClip             |:white\_check\_mark:|[LINK](https://huggingface.co/openai/clip-vit-base-patch32)               |
|InternVL2-4B         |:white\_check\_mark:|[LINK](https://huggingface.co/OpenGVLab/InternVL2-4B)                     |
|InternVL2-2B         |:white\_check\_mark:|[LINK](https://huggingface.co/OpenGVLab/InternVL2-2B)                     |
|MiniCPM-V-2_6        |:white\_check\_mark:|[LINK](https://huggingface.co/openbmb/MiniCPM-V-2_6)                      |
|Molmo-7B-D-0924      |:white\_check\_mark:|[LINK](https://huggingface.co/allenai/Molmo-7B-D-0924)                    |

如果您想要知道转换细节和源码，可以到本项目[models](./models)子目录查看各类模型部署细节。

如果您想看看演示效果，可以根据`Quick Start`内容操作即可。

如果您对我们的芯片感兴趣，也可以通过官网[SOPHGO](https://www.sophgo.com/)联系我们。

# Quick Start

如果您手上有BM1688或CV168AH的开发板，那么可以参考以下步骤跑通大语言模型，这里以Qwen1.5-1.8B为例。


## 跑通Demo

```
git clone https://github.com/sophgo/LLM-TPU_Lite.git
./run.sh --model qwen1.5
```


## 效果图
跑通后效果如下图所示

![](./assets/qwen-7b.png)

## Command Table

目前用于演示的模型，全部命令如下表所示

| Model           | Commnad                                     |
| :-------------- | :------------------------------------------ |
| ChatGLM3-6B     | ./run.sh --model chatglm3                   |
| Llama2-7B       | ./run.sh --model llama2                     |
| Qwen1.5-1.8B    | ./run.sh --model qwen1.5                    |
| Qwen2.5-1.5B    | ./run.sh --model qwen2.5                    |
| MiniCPM-2B      | ./run.sh --model minicpm                    |
| Phi-3-3.8B      | ./run.sh --model phi-3                      |
| Gemma2-2B       | ./run.sh --model gemma2                     |
| OpenClip        | ./run.sh --model openclip                   |
| InternVL2-2B    | ./run.sh --model internvl2                  |
| MiniCPM-V-2_6   | ./run.sh --model minicpmv2_6                |
| Molmo-7B-D-0924 | ./run.sh --model molmo-7b                   |



# 常见问题

## Q1：如果我的环境没有联网，那么怎么跑通大语言模型？

A：您可以先在联网的大机器上git clone本项目，之后运行 `./run.sh --model qwen1.5`

然后把LLM-TPU_Lite的全部文件拷贝到开发板上，必须要是全部文件，包括LLM-TPU_Lite/models，LLM-TPU_Lite/sg_llm

最后再在开发板上运行 `./run.sh --model qwen1.5`

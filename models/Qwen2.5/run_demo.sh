#!/bin/bash
set -ex

# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

model=qwen2.5-1.5b_int4_2048seq_1688_2core.bmodel

if [ ! -f "../../bmodels/${model}" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/${model}
  mv qwen2.5-1.5b_int4_2048seq_1688_2core.bmodel ../../bmodels
else
  echo "${model} Exists!"
fi


# run demo
cd python_demo && mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=~/.local/lib/python3.8/site-packages/pybind11 && make
cp *.so ../ && cd ..
python3 ./python_demo/chat.py --model ../../bmodels/${model} --tokenizer ../support/token_config

#!/bin/bash
set -ex

# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

model=qwen1.5-1.8b_bm1688_int4_2core.bmodel

if [ ! -f "../../bmodels/${model}" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/${model}
  mv qwen-7b_int4_1dev_none_addr.bmodel ../../bmodels
else
  echo "${model} Exists!"
fi


# run demo
# source /etc/profile.d/libsophon-bin-path.sh
# export LD_LIBRARY_PATH=$PWD/../libsophon-0.5.0/lib
python3 ./python_demo/chat.py --model ../../bmodels/${model} --tokenizer ./token_config

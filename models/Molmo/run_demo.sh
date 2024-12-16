#!/bin/bash
set -ex

# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

model=molmo-7b_int4_seq1024_384x384_1core.bmodel

if [ ! -f "../../bmodels/${model}" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/${model}
  mv molmo-7b_int4_seq1024_384x384_1core.bmodel ../../bmodels
else
  echo "${model} Exists!"
fi


# run demo
cd python_demo && mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=~/.local/lib/python3.8/site-packages/pybind11 && make
cp *.so ../ && cd ..
python3 pipeline.py --model ../../../bmodels/${model} --tokenizer ../support/processor_config --image_size 384 --image_path ./test.jpg 

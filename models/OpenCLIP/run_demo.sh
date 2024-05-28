#!/bin/bash
set -ex

# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

model=OpenClip_int4_2core.bmodel
pip3 install -r requirements.txt

if [ ! -f "../../bmodels/${model}" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/${model}
  mv OpenClip_int4_2core.bmodel ../../bmodels
else
  echo "${model} Exists!"
fi

if [ ! -f "./demo/demo.cpython-38-aarch64-linux-gnu.so" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. -DCMAKE_PREFIX_PATH=~/.local/lib/python3.8/site-packages/pybind11 && make 
  cp demo.cpython-38-aarch64-linux-gnu.so .. && cd ../..
else
  echo "File Exists!"
fi

python3 ./demo/demo.py --model_path ../../bmodels/${model} --config_path ./config

#!/bin/bash
set -ex

#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/internvl2-2b_bm1688_int4_2core.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/internvl2-2b_bm1688_int4_2core.bmodel
  mv internvl2-2b_bm1688_int4_2core.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi


if [ ! -f "./python_demo/*cpython*" ]; then
  pushd python_demo
  rm -rf build && mkdir build && cd build
  cmake .. && make
  cp *cpython* ..
  popd
else
  echo "chat.so exists!"
fi

# run demo
echo $PWD
python3 python_demo/pipeline.py --model ../../bmodels/internvl2-2b_bm1688_int4_2core.bmodel --tokenizer ./support/token_config_2b

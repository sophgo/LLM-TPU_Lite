#!/bin/bash
set -ex

#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/minicpm-2b_int4_2core.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/minicpm-2b_int4_2core.bmodel
  mv minicpm-2b_int4_2core.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi

if [ ! -f "./demo/minicpm" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp minicpm .. && cd ../..
else
  echo "minicpm file Exists!"
fi

# run demo
./demo/minicpm --model ../../bmodels/minicpm-2b_int4_2core.bmodel --tokenizer ./support/tokenizer.model

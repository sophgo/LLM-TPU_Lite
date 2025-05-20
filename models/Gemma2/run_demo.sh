#!/bin/bash
set -ex

#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/gemma2-2b_bm1688_int4_2core.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/gemma2-2b_bm1688_int4_2core.bmodel
  mv gemma2-2b_bm1688_int4_2core.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi

if [ ! -f "./demo/gemma2" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp gemma2 .. && cd ../..
else
  echo "gemma2 file Exists!"
fi

# run demo
./demo/gemma2 --model ../../bmodels/gemma2-2b_bm1688_int4_2core.bmodel --tokenizer ./support/tokenizer.model

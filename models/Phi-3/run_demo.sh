#!/bin/bash
set -ex

#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/phi-3_int4_2core.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/phi-3_int4_2core.bmodel
  mv phi-3_int4_2core.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi

if [ ! -f "./demo/phi-3" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp phi-3 .. && cd ../..
else
  echo "phi-3 file Exists!"
fi

# run demo
./demo/phi-3 --model ../../bmodels/phi-3_int4_2core.bmodel --tokenizer ./support/tokenizer.model

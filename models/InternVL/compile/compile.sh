#!/bin/bash
set -ex
models=
mode="int4"
folder="tmp"
mode_args=""
quantize_args="--quantize W4BF16"
name=""
num_layers=
out_model=$name.bmodel
num_core=""

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --mode)
            mode="$2"
            shift 2
            ;;
        --name)
            name="$2"
            shift 2
            ;;
        --num_core)
            num_core="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $key" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

if [ "$name" = "phi-3" ]; then
  num_layers=32
  echo "Compile Phi-3"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mphi-3\033[0m"
  exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8F16"
elif [ x$mode == x"f16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

out_model=$name'_'$mode'_'$num_core'core.bmodel'

outdir=${folder}/'embedding_'$num_core'core'
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ../onnx/embedding.onnx \
    --mlir embedding.mlir


model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1688 \
    --num_core $num_core \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ../onnx/embedding.onnx \
    --input_shapes [[1,1]] \
    --mlir embedding_cache.mlir


model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1688 \
    --num_core $num_core \
    --model embedding_cache.bmodel

rm *.npz

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd

echo $models

outdir='tmp/'$name'_bm1688_'$mode'/lm_head_'$num_core'core'
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ../../onnx/lm_head.onnx \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    $quantize_args \
    --quant_input \
    --quant_output \
    --chip bm1688 \
    --num_core $num_core \
    --model lm_head.bmodel

model_transform.py \
    --model_name greedy_head \
    --model_def ../../onnx/greedy_head.onnx \
    --mlir greedy_head.mlir

model_deploy.py \
    --mlir greedy_head.mlir \
    --chip bm1688 \
    --model greedy_head.bmodel

model_transform.py \
    --model_name penalty_sample_head \
    --model_def ../../onnx/penalty_sample_head.onnx \
    --mlir penalty_sample_head.mlir

model_deploy.py \
    --mlir penalty_sample_head.mlir \
    --chip bm1688 \
    --model penalty_sample_head.bmodel

rm *.npz

models=${models}${outdir}'/lm_head.bmodel '$outdir'/greedy_head.bmodel '$outdir'/penalty_sample_head.bmodel '
popd

echo $models

outdir='tmp/'$name'_bm1688_'$mode'/block_'$num_core'core'
mkdir -p $outdir

pushd $outdir
mkdir -p $outdir

for ((i=0; i<$num_layers; i++)); do

    model_transform.py \
        --model_name block_$i \
        --model_def ../../onnx/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1688 \
        --num_core $num_core \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ../../onnx/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1688 \
        --num_core $num_core \
        --addr_mode io_alone \
        --model block_cache_$i.bmodel

    rm *.npz

    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

model_transform.py \
    --model_name intern_vit \
    --model_def ./vit/onnx/vision_transformer.onnx \
    --input_shapes [[13,3,448,448]] \
    --mlir intern_vit.mlir \

model_deploy.py \
    --mlir intern_vit.mlir \
    --quantize BF16 \
    --processor bm1688 \
    --quant_output \
    --model intern_vit_bm1688_bf16.bmodel

model_tool --combine $models intern_vit_bm1688_bf16.bmodel -o $out_model

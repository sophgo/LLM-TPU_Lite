#!/bin/bash
# ./compile.sh --chip bm1688 --name gemma-2b
set -ex
models=
mode="int4"
folder="tmp"
quantize_args=""
chip_args=""
name="gemma-2b"
num_layers=
chip="bm1688"

onnx_dir=$PWD/tmp/onnx

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --mode)
        mode="$2"
        shift 2
        ;;
    --chip)
        chip="$2"
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

if [ "$name" = "gemma-2b" ]; then
  num_layers=18
  echo "Compile Gemma-2B"
elif [ "$name" = "gemma-7b" ]; then
  num_layers=28
  echo "Compile Gemma-7B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mgemma-2b|gemma-7b\033[0m"
  exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8BF16"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

if [ x$chip == x"bm1684x" ]; then
    chip_args="--chip bm1684x"
elif [ x$chip == x"bm1688" ]; then
    chip_args="--chip bm1688"
else
    echo "Error, unknown chip"
    exit 1
fi

folder='tmp/'$name'_'$chip'_'$mode'_'$num_core'core'
out_model=$name'_'$chip'_'$mode'_'$num_core'core.bmodel'


outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def $onnx_dir/embedding.onnx \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    $chip_args \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def $onnx_dir/embedding.onnx \
    --input_shapes [[1,1]] \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    $chip_args \
    --model embedding_cache.bmodel

rm *.npz *.onnx -f

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd

echo $models

outdir=${folder}/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def $onnx_dir/lm_head.onnx \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    $quantize_args \
    --quant_input \
    --quant_output \
    $chip_args \
    --num_core $num_core \
    --model lm_head.bmodel

rm *.npz *.onnx -f

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

outdir=${folder}/block
mkdir -p $outdir

pushd $outdir
mkdir -p $outdir

for ((i=0; i<$num_layers; i++)); do

    model_transform.py \
        --model_name block_$i \
        --model_def $onnx_dir/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        $chip_args \
        --num_core $num_core \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def $onnx_dir/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        $chip_args \
        --num_core $num_core \
        --addr_mode io_alone \
        --model block_cache_$i.bmodel

    rm *.npz *.onnx -f

    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

model_tool --combine $models -o $out_model

#!/bin/bash
# ./compile.sh --num_core 2
set -ex
models=""
mode=int4
mode_args=""
quantize_args=""
name="minicpmv2_6"

chip="bm1688"
num_layers=28
out_model=$name.bmodel

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

onnx_dir=$PWD/tmp/onnx
folder='tmp/'$name'_'$chip'_'$mode'_'$num_core'core'
out_model=$name'_'$chip'_'$mode'_'$num_core'core.bmodel'

# Compile Block
outdir=${folder}/block
mkdir -p $outdir
pushd $outdir

for ((i = 0; i < $num_layers; i++)); do

    rm *.npz *.onnx -f

    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

# convert embedding
outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir


rm *.npz *.onnx -f

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd

echo $models

# convert lm_head

outdir=${folder}/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ${onnx_dir}/lm_head.onnx \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    $quantize_args \
    --quant_input \
    --chip ${chip} \
    --num_core $num_core \
    --model lm_head.bmodel

rm *.npz *.onnx -f

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

# Compile VIT model
outdir=${folder}/vit
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name vision_encoder \
    --model_def ${onnx_dir}/vision_transformer.onnx \
    --mlir vision_encoder.mlir

model_deploy.py \
    --mlir vision_encoder.mlir \
    --quantize BF16 \
    --chip ${chip} \
    --num_core $num_core \
    --quant_output \
    --model vision_encoder_bf16.bmodel

rm *.npz *.onnx -f

models=${models}${outdir}'/vision_encoder_bf16.bmodel '

popd
echo $models

model_tool --combine $models -o $out_model

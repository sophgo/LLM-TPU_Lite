#!/bin/bash

set -ex
models=
mode="bf16"
folder="tmp"
mode_args=""
quantize_args="--quantize W4BF16"
num_layers=
out_model=""
num_core="1"

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --mode)
            mode="$2"
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
elif [ x$mode == x"f16" ]; then
    quantize_args="--quantize F16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

out_model='OpenClip_'$mode'_'$num_core'core.bmodel'

outdir=${folder}/'OpenClip_'$num_core'core'
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name OpenClip \
    --input_shape [[2,77],[2,77],[1,3,224,224]] \
    --model_def ../onnx/openclip.onnx \
    --mlir openclip.mlir
    # --test_input ../../input_ref.npz \
    # --test_result ../../output_top.npz


model_deploy.py \
    --mlir openclip.mlir \
    $quantize_args \
    --chip bm1688 \
    --num_core $num_core \
    --model ${out_model}
    # --test_input OpenClip_in_f32.npz \
    # --test_reference ../../output_top.npz

rm *.npz *.onnx -f

popd

cp $outdir/${out_model} .

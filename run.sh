#!/bin/bash
set -ex

# Args
parse_args() {
    while [[ $# -gt 0 ]]; do
        key="$1"

        case $key in
            --model)
                model="$2"
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
}

# Mapping
declare -A model_to_demo=(
    ["chatglm3"]="ChatGLM3"
    ["llama2"]="Llama2"
    ["qwen1.5"]="Qwen1.5"
    ["minicpm"]="MiniCPM"
    ["phi-3"]="Phi-3"
    ["gemma2"]="Gemma2"
    ["openclip"]="OpenCLIP"
    ["internvl2"]="InternVL2"
    ["minicpmv2_6"]="MiniCPM-V-2_6"
)

# Function to validate model name
validate_model() {
    local model="$1"
    if [[ ! ${model_to_demo[$model]} ]]; then
        echo -e "Error: Invalid name $model, the input name must be \033[31m$(printf "%s|" "${!model_to_demo[@]}" | sed 's/|$//')\033[0m" >&2
        return 1
    fi
    return 0
}

# Process Args
parse_args "$@"

# Check Model Name
validate_model "$model" || exit 1

# Compile
pushd "./models/${model_to_demo[$model]}"
./run_demo.sh
popd

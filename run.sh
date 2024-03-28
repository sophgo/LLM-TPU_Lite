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
    ["qwen1.5"]="Qwen"
)

# Process Args
parse_args "$@"

# Check Model Name
if [[ ! ${model_to_demo[$model]} ]]; then
    >&2 echo -e "Error: Invalid name $model, the input name must be \033[31mchatglm3|llama2|qwen1.5\033[0m"
    exit 1
fi

# Compile
pushd "./models/${model_to_demo[$model]}"
./run_demo.sh
popd

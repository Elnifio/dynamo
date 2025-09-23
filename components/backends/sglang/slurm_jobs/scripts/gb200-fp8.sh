#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Function to print usage
print_usage() {
    echo "Usage: $0 <mode>"
    echo "  mode: prefill or decode"
    echo ""
    echo "Examples:"
    echo "  $0 prefill"
    echo "  $0 decode"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 1 ]; then
    echo "Error: Expected 1 argument, got $#"
    print_usage
fi

# Parse arguments
mode=$1

# Validate mode argument
if [ "$mode" != "prefill" ] && [ "$mode" != "decode" ]; then
    echo "Error: mode must be 'prefill' or 'decode', got '$mode'"
    print_usage
fi

echo "Mode: $mode"
echo "Command: dynamo"

# Check if required environment variables are set
if [ -z "$HOST_IP" ]; then
    echo "Error: HOST_IP environment variable is not set"
    exit 1
fi

if [ -z "$PORT" ]; then
    echo "Error: PORT environment variable is not set"
    exit 1
fi

if [ -z "$TOTAL_GPUS" ]; then
    echo "Error: TOTAL_GPUS environment variable is not set"
    exit 1
fi

if [ -z "$RANK" ]; then
    echo "Error: RANK environment variable is not set"
    exit 1
fi

if [ -z "$TOTAL_NODES" ]; then
    echo "Error: TOTAL_NODES environment variable is not set"
    exit 1
fi

if [ -z "$USE_INIT_LOCATIONS" ]; then
    echo "Error: USE_INIT_LOCATIONS environment variable is not set"
    exit 1
fi

# Construct command based on mode
if [ "$mode" = "prefill" ]; then
    # GB200 dynamo prefill command
    set -x
    # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048 \

    if [[ "${USE_INIT_LOCATIONS,,}" == "true" ]]; then command_suffix="--init-expert-location /configs/prefill_dsr1-0528_in1000out1000_num40000.json"; fi

    export SGL_ENABLE_JIT_DEEPGEMM=false
    export SGLANG_ENABLE_FLASHINFER_GEMM=true
    command=(
        python3 -m dynamo.sglang.worker
        --tokenizer-path "/model/" --model-path "/model/" --served-model-name deepseek-ai/DeepSeek-R1
        --host 0.0.0.0 --port 8000
        --dist-init-addr $HOST_IP:$PORT
        --nnodes $TOTAL_NODES --node-rank $RANK
        --tensor-parallel-size=$TOTAL_GPUS --data-parallel-size=1
        --disaggregation-mode prefill --disaggregation-bootstrap-port 30001
        --trust-remote-code --skip-tokenizer-init
        --disable-radix-cache
        --max-running-requests 512
        --chunked-prefill-size 32768
        --mem-fraction-static 0.7
        --cuda-graph-max-bs 512
        --max-prefill-tokens 32768
        --kv-cache-dtype fp8_e4m3
        --quantization fp8
        --attention-backend trtllm_mla
        --stream-interval 10
        --enable-flashinfer-trtllm-moe
        --scheduler-recv-interval 10
    )
    
    ${command[@]}

elif [ "$mode" = "decode" ]; then
    set -x
    command_suffix=""
    if [[ "${USE_INIT_LOCATIONS,,}" == "true" ]]; then command_suffix="--init-expert-location /configs/decode_dsr1-0528_loadgen_in1024out1024_num2000_2p12d.json"; fi

    export SGL_ENABLE_JIT_DEEPGEMM=false
    export SGLANG_ENABLE_FLASHINFER_GEMM=true
    command=(
        python3 -m dynamo.sglang.worker
        --tokenizer-path "/model/" --model-path "/model/" --served-model-name deepseek-ai/DeepSeek-R1
        --host 0.0.0.0 --port 8000
        --dist-init-addr $HOST_IP:$PORT
        --nnodes $TOTAL_NODES --node-rank $RANK
        --tensor-parallel-size=$TOTAL_GPUS --data-parallel-size=1
        --disaggregation-mode decode --disaggregation-bootstrap-port 30001
        --trust-remote-code --skip-tokenizer-init
        --disable-radix-cache
        --max-running-requests 512
        --chunked-prefill-size 32768
        --mem-fraction-static 0.7
        --cuda-graph-max-bs 512
        --max-prefill-tokens 32768
        --kv-cache-dtype fp8_e4m3
        --quantization fp8
        --attention-backend trtllm_mla
        --stream-interval 10
        --enable-flashinfer-trtllm-moe
        --scheduler-recv-interval 10
    )
    
    ${command[@]}
fi

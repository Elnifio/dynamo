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

DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
MC_TE_METRIC=true \
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
MC_FORCE_MNNVL=1 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m sglang.launch_server \
--tokenizer-path "/model/" \
--trust-remote-code \
--disable-radix-cache \
--max-running-requests 512 \
--chunked-prefill-size 32768 \
--mem-fraction-static 0.7 \
--cuda-graph-max-bs 512 \
--max-prefill-tokens 32768 \
--kv-cache-dtype fp8_e4m3 \
--attention-backend trtllm_mla \
--stream-interval 10 \
--enable-flashinfer-trtllm-moe \
--model-path="/model/" \
--host 0.0.0.0 --port 8000 \
--tensor-parallel-size=8 --data-parallel-size=1

#!/bin/bash
# TensorFlow GPU テスト実行スクリプト

# スクリプトのディレクトリに移動
cd "$(dirname "$0")"

# 仮想環境のPythonバージョンを取得
PYTHON_VERSION=$(./venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# NVIDIA CUDAライブラリのパスを設定
NVIDIA_LIB_PATH="./venv/lib/python${PYTHON_VERSION}/site-packages/nvidia"

# 各NVIDIAライブラリのlibディレクトリを検索してLD_LIBRARY_PATHに追加
CUDA_LIB_PATHS=""
for lib_dir in ${NVIDIA_LIB_PATH}/*/lib; do
    if [ -d "$lib_dir" ]; then
        if [ -z "$CUDA_LIB_PATHS" ]; then
            CUDA_LIB_PATHS="$lib_dir"
        else
            CUDA_LIB_PATHS="$CUDA_LIB_PATHS:$lib_dir"
        fi
    fi
done

# LD_LIBRARY_PATHを設定
if [ -n "$CUDA_LIB_PATHS" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIB_PATHS:$LD_LIBRARY_PATH"
    echo "CUDAライブラリパスを設定しました:"
    echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep nvidia | while read line; do
        echo "  - $line"
    done
    echo ""
fi

# TensorFlowテストを実行
echo "TensorFlowテストを実行します..."
echo "===================="
./venv/bin/python cuda_tensorflow_teset.py

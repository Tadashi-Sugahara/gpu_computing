#!/bin/bash
# OpenCV GPU vs CPU 比較スクリプトの実行用スクリプト
# venvのPythonで実行し、CUDAライブラリパスを設定します

# 現在のディレクトリに移動
cd "$(dirname "$0")"

VENV_PYTHON="$(pwd)/venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
	echo "エラー: venv の Python が見つかりません: $VENV_PYTHON"
	exit 1
fi

# NVIDIA CUDAライブラリをLD_LIBRARY_PATHに追加
NVIDIA_LIB_DIRS=$(find "$(pwd)/venv/lib" -type d -path "*/site-packages/nvidia/*/lib" 2>/dev/null | tr '\n' ':')
if [ -n "$NVIDIA_LIB_DIRS" ]; then
	NVIDIA_LIB_DIRS="${NVIDIA_LIB_DIRS%:}"
	if [ -n "$LD_LIBRARY_PATH" ]; then
		export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS:$LD_LIBRARY_PATH"
	else
		export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS"
	fi
fi

echo "============================================"
echo "  GPU vs CPU 比較スクリプト実行"
echo "============================================"
echo ""
echo "使用Python: $VENV_PYTHON"
echo ""

"$VENV_PYTHON" opencv_gpu_test.py

echo ""
echo "実行完了"

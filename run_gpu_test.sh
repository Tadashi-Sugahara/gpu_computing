#!/bin/bash
# OpenCV GPU vs CPU 比較スクリプトの実行用スクリプト
# venvを完全に無効化してシステムのPython3で実行します

# 現在のディレクトリに移動
cd "$(dirname "$0")"

# すべての仮想環境変数をクリア
unset VIRTUAL_ENV
unset PYTHONHOME
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "/venv/" | tr '\n' ':' | sed 's/:$//')

# システムのPython3で直接実行
echo "============================================"
echo "  GPU vs CPU 比較スクリプト実行"
echo "============================================"
echo ""
echo "使用Python: /usr/bin/python3"
echo ""

/usr/bin/python3 opencv_gpu_test.py

echo ""
echo "実行完了"

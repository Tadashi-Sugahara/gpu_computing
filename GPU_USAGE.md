# GPUの使い方（自動設定版）

## 概要

仮想環境の`activate`スクリプトを修正し、**activateするだけで自動的にGPUが使えるようになりました**。

## 使い方

### ターミナルから実行

```bash
# 1. 仮想環境をactivate
source venv/bin/activate

# 2. あとは普通にPythonを実行
python cuda_tensorflow_teset.py

# または他のスクリプト
python diagnose_gpu.py
python cupy_test.py
python opencv_gpu_test.py
```

### VS Codeから実行

1. ファイルを開く（例：`cuda_tensorflow_teset.py`）
2. 右上の実行ボタン（▶）をクリック、またはF5キーを押す
3. GPUが自動的に使われます！

### Jupyter Notebookから実行

```python
import tensorflow as tf

# GPUが自動的に認識されます
print(tf.config.list_physical_devices('GPU'))
```

## 技術的な詳細

### 何が変更されたか

`venv/bin/activate`スクリプトに以下を追加：

```bash
# activate時に実行される
_OLD_VIRTUAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
CUDA_LIB_PATHS=""
for lib_dir in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$lib_dir" ]; then
        CUDA_LIB_PATHS="$CUDA_LIB_PATHS:$lib_dir"
    fi
done
export LD_LIBRARY_PATH="$CUDA_LIB_PATHS:$LD_LIBRARY_PATH"

# deactivate時に元に戻す
deactivate() {
    ...
    LD_LIBRARY_PATH="${_OLD_VIRTUAL_LD_LIBRARY_PATH:-}"
    ...
}
```

### 設定されるライブラリパス

```
./venv/lib/python3.12/site-packages/nvidia/cublas/lib
./venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib
./venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
./venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
./venv/lib/python3.12/site-packages/nvidia/cudnn/lib
./venv/lib/python3.12/site-packages/nvidia/cufft/lib
./venv/lib/python3.12/site-packages/nvidia/curand/lib
./venv/lib/python3.12/site-packages/nvidia/cusolver/lib
./venv/lib/python3.12/site-packages/nvidia/cusparse/lib
./venv/lib/python3.12/site-packages/nvidia/nccl/lib
./venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib
```

## 他のプロジェクトで同じ設定をする方法

新しいプロジェクトで同じ設定をする場合：

### 方法1: 仮想環境のactivateスクリプトを手動で編集

`venv/bin/activate`を開き、上記のコードを追加します。

### 方法2: 自動設定スクリプト

```bash
#!/bin/bash
# setup_gpu_venv.sh - 仮想環境にGPU設定を追加

ACTIVATE_FILE="venv/bin/activate"

if [ ! -f "$ACTIVATE_FILE" ]; then
    echo "Error: $ACTIVATE_FILE が見つかりません"
    exit 1
fi

# deactivate関数にLD_LIBRARY_PATHの復元を追加
sed -i '/unset _OLD_VIRTUAL_PYTHONHOME/a\    if [ -n "${_OLD_VIRTUAL_LD_LIBRARY_PATH:-}" ] ; then\n        LD_LIBRARY_PATH="${_OLD_VIRTUAL_LD_LIBRARY_PATH:-}"\n        export LD_LIBRARY_PATH\n        unset _OLD_VIRTUAL_LD_LIBRARY_PATH\n    fi' "$ACTIVATE_FILE"

# CUDA library pathsの設定を追加
cat >> "$ACTIVATE_FILE" << 'EOF'

# Setup CUDA library paths for TensorFlow GPU support
_OLD_VIRTUAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
CUDA_LIB_PATHS=""
for lib_dir in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$lib_dir" ]; then
        if [ -z "$CUDA_LIB_PATHS" ]; then
            CUDA_LIB_PATHS="$lib_dir"
        else
            CUDA_LIB_PATHS="$CUDA_LIB_PATHS:$lib_dir"
        fi
    fi
done
if [ -n "$CUDA_LIB_PATHS" ]; then
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        LD_LIBRARY_PATH="$CUDA_LIB_PATHS:$LD_LIBRARY_PATH"
    else
        LD_LIBRARY_PATH="$CUDA_LIB_PATHS"
    fi
    export LD_LIBRARY_PATH
fi
EOF

echo "GPU設定を追加しました: $ACTIVATE_FILE"
```

使い方：
```bash
chmod +x setup_gpu_venv.sh
./setup_gpu_venv.sh
```

## 確認方法

### 環境変数の確認

```bash
source venv/bin/activate
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep nvidia
```

### GPUが認識されているか確認

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

期待される出力：
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## トラブルシューティング

### GPUが認識されない場合

1. 仮想環境を再activate
   ```bash
   deactivate
   source venv/bin/activate
   ```

2. LD_LIBRARY_PATHを確認
   ```bash
   echo $LD_LIBRARY_PATH
   ```

3. cuDNNライブラリが存在するか確認
   ```bash
   ls venv/lib/python*/site-packages/nvidia/cudnn/lib/
   ```

4. 診断ツールを実行
   ```bash
   python diagnose_gpu.py
   ```

### VS Codeで認識されない場合

1. VS Codeを再起動
2. Python環境を再選択
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - `./venv/bin/python`を選択

## 利点

✅ シェルスクリプトを実行する必要がない  
✅ `python`コマンドで直接実行できる  
✅ VS CodeのRun/Debugがそのまま使える  
✅ Jupyter Notebookでも動作する  
✅ 環境変数の設定を忘れない  
✅ deactivate時に自動的に元に戻る  

## 関連ファイル

- `venv/bin/activate` - 修正済み（GPU自動設定あり）
- `cuda_tensorflow_teset.py` - TensorFlowのGPUテスト
- `diagnose_gpu.py` - GPU環境診断ツール
- `run_tensorflow_gpu_test.sh` - 旧実行スクリプト（もう不要）

# TensorFlow GPU 使用ガイド

## 問題と解決策

### 問題
TensorFlowがGPUを認識できず、以下のエラーが発生していました：
```
Cannot dlopen some GPU libraries. 
GPUが検出されませんでした
```

### 原因
- cuDNNライブラリは`pip`でインストール済みだった
- しかし、TensorFlowが動的ライブラリを見つけられなかった
- `LD_LIBRARY_PATH`環境変数が設定されていなかった

### 解決策
CUDAライブラリのパスを`LD_LIBRARY_PATH`に追加する必要がありました。

## システム環境

- **GPU**: NVIDIA GeForce RTX 2060
- **ドライバー**: 580.126.09
- **CUDA**: 12.0
- **cuDNN**: 9.19.0 (pip経由)
- **TensorFlow**: 2.20.0

## インストールされているGPUライブラリ

```
nvidia-cudnn-cu12==9.19.0.56
nvidia-cublas-cu12==12.9.1.4
cupy-cuda12x==13.6.0
```

## 実行方法

### 推奨方法（ライブラリパス自動設定）
```bash
./run_tensorflow_gpu_test.sh
```

このスクリプトが以下を自動実行します：
1. 必要なCUDAライブラリパスを検出
2. `LD_LIBRARY_PATH`を設定
3. TensorFlowテストを実行

### 手動実行する場合
```bash
# ライブラリパスを設定
export LD_LIBRARY_PATH=$(find ./venv/lib/python3.12/site-packages/nvidia/*/lib -type d | tr '\n' ':')$LD_LIBRARY_PATH

# Pythonスクリプト実行
./venv/bin/python cuda_tensorflow_teset.py
```

## パフォーマンス結果

テスト実行の結果：

| 演算タイプ | CPU時間 | GPU時間 | 高速化率 |
|----------|--------|--------|---------|
| 行列乗算 (5000x5000) | 0.34秒 | 0.0006秒 | **571倍** |
| 畳み込み演算 (CNN) | 0.09秒 | 0.006秒 | **16倍** |
| NN学習 | - | 2.48秒 | - |

## 診断ツール

GPU環境の問題を診断するには：
```bash
./venv/bin/python diagnose_gpu.py
```

このツールが以下をチェックします：
- NVIDIAドライバーの状態
- CUDA Toolkitのインストール
- cuDNNライブラリの有無
- TensorFlowのGPU認識
- その他のGPUライブラリ（CuPy, PyTorchなど）

## トラブルシューティング

### GPUが検出されない場合

1. **NVIDIAドライバーの確認**
   ```bash
   nvidia-smi
   ```

2. **CUDA Toolkitの確認**
   ```bash
   nvcc --version
   ```

3. **cuDNNのインストール確認**
   ```bash
   pip list | grep cudnn
   # または
   ldconfig -p | grep cudnn
   ```

4. **ライブラリパスの確認**
   ```bash
   echo $LD_LIBRARY_PATH
   ```

### cuDNNがない場合

**オプション1: pipでインストール（推奨）**
```bash
pip install nvidia-cudnn-cu12
```

**オプション2: aptでインストール**
```bash
sudo apt update
sudo apt install libcudnn9-cuda-12
```

**オプション3: NVIDIA公式サイトから**
1. https://developer.nvidia.com/cudnn にアクセス
2. NVIDIAアカウントでログイン
3. CUDA 12.x用のcuDNN 9.xをダウンロード
4. インストール手順に従う

インストール後：
```bash
sudo ldconfig
./venv/bin/python diagnose_gpu.py
```

## 今後のプロジェクトで同じ問題を回避する方法

### 方法1: 起動スクリプトを使用
上記の`run_tensorflow_gpu_test.sh`のようなスクリプトを作成

### 方法2: .bashrcに追加
```bash
echo 'export LD_LIBRARY_PATH=$HOME/Python/cuda/gpu_computing/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 方法3: sitecustomize.pyを使用
仮想環境のsite-packagesに`sitecustomize.py`を作成して、Pythonの起動時に自動設定

### 方法4: システム全体にcuDNNをインストール
```bash
sudo apt install libcudnn9-cuda-12
sudo ldconfig
```

## 参考リンク

- [TensorFlow GPU サポート](https://www.tensorflow.org/install/gpu)
- [CUDA Toolkit ダウンロード](https://developer.nvidia.com/cuda-downloads)
- [cuDNN ダウンロード](https://developer.nvidia.com/cudnn)

## 関連ファイル

- `cuda_tensorflow_teset.py` - TensorFlowのGPUテストスクリプト
- `run_tensorflow_gpu_test.sh` - 実行スクリプト（ライブラリパス自動設定）
- `diagnose_gpu.py` - GPU環境診断ツール
- `cupy_test.py` - CuPyのGPUテスト
- `opencv_gpu_test.py` - OpenCVのGPUテスト

## 注意事項

- `LD_LIBRARY_PATH`の設定は、Pythonプロセスを起動する**前**に行う必要があります
- Pythonスクリプト内で`os.environ['LD_LIBRARY_PATH']`を設定しても、既に起動したプロセスには影響しません
- 仮想環境ごとにライブラリパスが異なるため、パスの設定に注意してください

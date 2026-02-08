#!/usr/bin/env python3
"""GPU環境の診断スクリプト"""

import sys
import os
import subprocess

def check_nvidia_driver():
    """NVIDIAドライバーの確認"""
    print("=" * 70)
    print("1. NVIDIAドライバーの確認")
    print("=" * 70)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line or 'CUDA Version' in line:
                    print(line.strip())
            print("✓ NVIDIAドライバーが正常にインストールされています")
        else:
            print("✗ nvidia-smiの実行に失敗しました")
    except FileNotFoundError:
        print("✗ nvidia-smiが見つかりません。NVIDIAドライバーがインストールされていない可能性があります")

def check_cuda_installation():
    """CUDA Toolkitの確認"""
    print("\n" + "=" * 70)
    print("2. CUDA Toolkitの確認")
    print("=" * 70)
    
    # nvccの確認
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line.lower():
                    print(f"✓ {line.strip()}")
        else:
            print("✗ nvccの実行に失敗しました")
    except FileNotFoundError:
        print("✗ nvccが見つかりません。CUDA Toolkitがインストールされていないか、PATHが設定されていません")
    
    # CUDA関連の環境変数
    print("\n環境変数:")
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']
    for var in cuda_vars:
        value = os.environ.get(var, '(未設定)')
        print(f"  {var}: {value}")

def check_cudnn():
    """cuDNNライブラリの確認"""
    print("\n" + "=" * 70)
    print("3. cuDNNライブラリの確認")
    print("=" * 70)
    
    try:
        result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
        cudnn_libs = [line for line in result.stdout.split('\n') if 'cudnn' in line.lower()]
        
        if cudnn_libs:
            print("✓ cuDNNライブラリが見つかりました:")
            for lib in cudnn_libs[:5]:  # 最初の5つのみ表示
                print(f"  {lib.strip()}")
        else:
            print("✗ cuDNNライブラリが見つかりません")
            print("\n解決方法:")
            print("  1. NVIDIA公式サイトからcuDNNをダウンロード: https://developer.nvidia.com/cudnn")
            print("  2. またはaptでインストール:")
            print("     sudo apt install libcudnn9-dev")
    except Exception as e:
        print(f"✗ エラー: {e}")

def check_tensorflow():
    """TensorFlowの確認"""
    print("\n" + "=" * 70)
    print("4. TensorFlowの確認")
    print("=" * 70)
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlowバージョン: {tf.__version__}")
        print(f"  CUDAサポート: {tf.test.is_built_with_cuda()}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ 検出されたGPU数: {len(gpus)}")
            for gpu in gpus:
                print(f"  {gpu}")
        else:
            print("✗ GPUが検出されませんでした")
            print("\n考えられる原因:")
            print("  - cuDNNがインストールされていない")
            print("  - TensorFlowとCUDA/cuDNNのバージョンが互換性がない")
            print("  - LD_LIBRARY_PATHが正しく設定されていない")
    except ImportError:
        print("✗ TensorFlowがインストールされていません")
    except Exception as e:
        print(f"✗ エラー: {e}")

def check_other_gpu_libraries():
    """その他のGPUライブラリの確認"""
    print("\n" + "=" * 70)
    print("5. その他のGPUライブラリの確認")
    print("=" * 70)
    
    libraries = {
        'CuPy': 'cupy',
        'PyTorch': 'torch',
        'JAX': 'jax'
    }
    
    for name, module in libraries.items():
        try:
            lib = __import__(module)
            version = getattr(lib, '__version__', '不明')
            print(f"✓ {name} バージョン: {version}")
            
            # GPU利用可能性の確認
            if module == 'cupy':
                try:
                    import cupy as cp
                    print(f"  GPU利用可能: {cp.cuda.is_available()}")
                except:
                    pass
            elif module == 'torch':
                try:
                    import torch
                    print(f"  CUDA利用可能: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        print(f"  GPUデバイス数: {torch.cuda.device_count()}")
                except:
                    pass
        except ImportError:
            print(f"- {name}: インストールされていません")

def print_recommendations():
    """推奨事項"""
    print("\n" + "=" * 70)
    print("6. 推奨事項")
    print("=" * 70)
    
    print("""
TensorFlowでGPUを使用するための要件:
1. NVIDIAドライバー (インストール済み)
2. CUDA Toolkit (インストール済み: 12.0)
3. cuDNN (未インストール) ← これが必要です

cuDNNのインストール方法:

オプション1: aptでインストール（推奨）
  sudo apt update
  sudo apt install libcudnn9-cuda-12

オプション2: pipでインストール（TensorFlow用の簡易版）
  pip install nvidia-cudnn-cu12

オプション3: NVIDIA公式からダウンロード
  1. https://developer.nvidia.com/cudnn にアクセス
  2. NVIDIAアカウントでログイン
  3. CUDA 12.x用のcuDNN 9.xをダウンロード
  4. インストール手順に従う

インストール後、以下を実行してください:
  sudo ldconfig
  python diagnose_gpu.py  # 再診断
""")

if __name__ == "__main__":
    print("\nGPU環境診断ツール")
    print("=" * 70)
    
    check_nvidia_driver()
    check_cuda_installation()
    check_cudnn()
    check_tensorflow()
    check_other_gpu_libraries()
    print_recommendations()
    
    print("\n" + "=" * 70)
    print("診断完了")
    print("=" * 70)

# OpenCV GPU処理のデモ（カメラ不要版）

import cv2
import numpy as np
import time

def process_cpu(frame):
    """CPU上で画像処理を実行"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def process_gpu(frame):
    """GPU上で画像処理を実行"""
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
    gaussian_filter = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC1, cv2.CV_8UC1, (15, 15), 0
    )
    gpu_blurred = gaussian_filter.apply(gpu_gray)
    canny_detector = cv2.cuda.createCannyEdgeDetector(50, 150)
    gpu_edges = canny_detector.detect(gpu_blurred)
    edges = gpu_edges.download()
    return edges

def main():
    print("=== OpenCV GPU処理デモ ===\n")
    
    # GPU確認
    gpu_available = False
    if hasattr(cv2, 'cuda'):
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_available = True
                print(f"✓ CUDA対応デバイス数: {cv2.cuda.getCudaEnabledDeviceCount()}")
                print(f"✓ 使用中のデバイス: {cv2.cuda.getDevice()}")
            else:
                print("⚠ CUDA対応のデバイスが見つかりません")
        except Exception as e:
            print(f"⚠ CUDA初期化エラー: {e}")
    else:
        print("⚠ OpenCVにCUDAモジュールがありません")
    
    if not gpu_available:
        print("→ CPU処理のみで実行します")
    
    print(f"\nOpenCV Version: {cv2.__version__}")
    print(f"Build Information:")
    print(f"  CUDA: {cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False}")
    
    # テスト画像を生成
    print("\n=== パフォーマンステスト ===")
    print("テスト画像サイズ: 1280x720")
    
    # ランダムな画像を生成
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # CPU処理のベンチマーク
    cpu_times = []
    print("\nCPU処理をテスト中...", end="", flush=True)
    for i in range(50):
        start_time = time.time()
        _ = process_cpu(test_frame)
        cpu_time = (time.time() - start_time) * 1000
        cpu_times.append(cpu_time)
        if (i + 1) % 10 == 0:
            print(f" {i+1}", end="", flush=True)
    print(" 完了")
    
    # GPU処理のベンチマーク（利用可能な場合）
    gpu_times = []
    if gpu_available:
        print("GPU処理をテスト中...", end="", flush=True)
        # ウォームアップ
        for _ in range(5):
            _ = process_gpu(test_frame)
        
        for i in range(50):
            start_time = time.time()
            _ = process_gpu(test_frame)
            gpu_time = (time.time() - start_time) * 1000
            gpu_times.append(gpu_time)
            if (i + 1) % 10 == 0:
                print(f" {i+1}", end="", flush=True)
        print(" 完了")
    
    # 結果を表示
    print("\n=== ベンチマーク結果 ===")
    print(f"CPU処理:")
    print(f"  平均: {np.mean(cpu_times):.2f}ms")
    print(f"  最小: {np.min(cpu_times):.2f}ms")
    print(f"  最大: {np.max(cpu_times):.2f}ms")
    
    if gpu_times:
        print(f"\nGPU処理:")
        print(f"  平均: {np.mean(gpu_times):.2f}ms")
        print(f"  最小: {np.min(gpu_times):.2f}ms")
        print(f"  最大: {np.max(gpu_times):.2f}ms")
        
        speedup = np.mean(cpu_times) / np.mean(gpu_times)
        print(f"\n高速化率: {speedup:.2f}x")
        
        if speedup > 1:
            print(f"✓ GPUはCPUより{speedup:.1f}倍高速です！")
        else:
            print("⚠ この処理ではGPUの利点がありません")
    
    print("\n=== CUDA対応OpenCVのインストール方法 ===")
    if not gpu_available:
        print("\nCUDA対応OpenCVを使用するには:")
        print("1. opencv-contribをソースからビルド（推奨）")
        print("   - CUDAツールキットが必要")
        print("   - ビルドに1-2時間かかる場合があります")
        print("\n2. または、事前ビルド版を探す")
        print("   - pip install opencv-contrib-python")
        print("   ※ただし、CUDA対応版は限定的")

if __name__ == "__main__":
    main()

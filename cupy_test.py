# CupyとNumpyの基本的な操作を比較するコード
import numpy as np
import cupy as cp
import time

def compare_array_operations():
    """配列の基本操作を比較"""
    print("=" * 50)
    print("配列の基本操作の比較")
    print("=" * 50)
    
    size = 80000000  # 1000万要素
    
    # Numpy配列の作成と操作
    print("\n[NumPy]")
    start = time.time()
    np_array1 = np.random.rand(size)
    np_array2 = np.random.rand(size)
    np_result = np_array1 + np_array2
    np_time = time.time() - start
    print(f"配列サイズ: {size:,}")
    print(f"実行時間: {np_time:.4f}秒")
    print(f"結果のサンプル: {np_result[:5]}")
    
    # Cupy配列の作成と操作
    print("\n[CuPy]")
    start = time.time()
    cp_array1 = cp.random.rand(size)
    cp_array2 = cp.random.rand(size)
    cp_result = cp_array1 + cp_array2
    cp.cuda.Stream.null.synchronize()  # GPU処理の完了を待つ
    cp_time = time.time() - start
    print(f"配列サイズ: {size:,}")
    print(f"実行時間: {cp_time:.4f}秒")
    print(f"結果のサンプル: {cp.asnumpy(cp_result[:5])}")
    print(f"\n高速化率: {np_time / cp_time:.2f}倍")

def compare_matrix_operations():
    """行列演算を比較"""
    print("\n" + "=" * 50)
    print("行列演算の比較")
    print("=" * 50)
    
    size = 2000
    
    # Numpy行列乗算
    print("\n[NumPy]")
    start = time.time()
    np_matrix1 = np.random.rand(size, size)
    np_matrix2 = np.random.rand(size, size)
    np_result = np.dot(np_matrix1, np_matrix2)
    np_time = time.time() - start
    print(f"行列サイズ: {size}x{size}")
    print(f"実行時間: {np_time:.4f}秒")
    print(f"結果のサンプル:\n{np_result[:2, :2]}")
    
    # Cupy行列乗算
    print("\n[CuPy]")
    start = time.time()
    cp_matrix1 = cp.random.rand(size, size)
    cp_matrix2 = cp.random.rand(size, size)
    cp_result = cp.dot(cp_matrix1, cp_matrix2)
    cp.cuda.Stream.null.synchronize()
    cp_time = time.time() - start
    print(f"行列サイズ: {size}x{size}")
    print(f"実行時間: {cp_time:.4f}秒")
    print(f"結果のサンプル:\n{cp.asnumpy(cp_result[:2, :2])}")
    print(f"\n高速化率: {np_time / cp_time:.2f}倍")

def compare_statistical_operations():
    """統計演算を比較"""
    print("\n" + "=" * 50)
    print("統計演算の比較")
    print("=" * 50)
    
    size = 100000000  # 1億要素
    
    # Numpy統計演算
    print("\n[NumPy]")
    start = time.time()
    np_array = np.random.rand(size)
    np_mean = np.mean(np_array)
    np_std = np.std(np_array)
    np_max = np.max(np_array)
    np_min = np.min(np_array)
    np_time = time.time() - start
    print(f"配列サイズ: {size:,}")
    print(f"平均: {np_mean:.6f}")
    print(f"標準偏差: {np_std:.6f}")
    print(f"最大値: {np_max:.6f}")
    print(f"最小値: {np_min:.6f}")
    print(f"実行時間: {np_time:.4f}秒")
    
    # Cupy統計演算
    print("\n[CuPy]")
    start = time.time()
    cp_array = cp.random.rand(size)
    cp_mean = cp.mean(cp_array)
    cp_std = cp.std(cp_array)
    cp_max = cp.max(cp_array)
    cp_min = cp.min(cp_array)
    cp.cuda.Stream.null.synchronize()
    cp_time = time.time() - start
    print(f"配列サイズ: {size:,}")
    print(f"平均: {float(cp_mean):.6f}")
    print(f"標準偏差: {float(cp_std):.6f}")
    print(f"最大値: {float(cp_max):.6f}")
    print(f"最小値: {float(cp_min):.6f}")
    print(f"実行時間: {cp_time:.4f}秒")
    print(f"\n高速化率: {np_time / cp_time:.2f}倍")

def compare_memory_transfer():
    """メモリ転送時間を測定"""
    print("\n" + "=" * 50)
    print("メモリ転送時間の測定")
    print("=" * 50)
    
    size = 10000000
    
    # CPUからGPUへの転送
    print("\n[CPU → GPU]")
    np_array = np.random.rand(size)
    start = time.time()
    cp_array = cp.asarray(np_array)
    cp.cuda.Stream.null.synchronize()
    cpu_to_gpu_time = time.time() - start
    print(f"転送時間: {cpu_to_gpu_time:.4f}秒")
    
    # GPUからCPUへの転送
    print("\n[GPU → CPU]")
    start = time.time()
    result_np = cp.asnumpy(cp_array)
    gpu_to_cpu_time = time.time() - start
    print(f"転送時間: {gpu_to_cpu_time:.4f}秒")

if __name__ == "__main__":
    print("CuPyとNumPyのパフォーマンス比較\n")
    
    # GPU情報を表示
    print(f"使用GPU: {cp.cuda.Device().compute_capability}")
    print(f"GPUメモリ: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB\n")
    
    # 各種比較を実行
    compare_array_operations()
    compare_matrix_operations()
    compare_statistical_operations()
    compare_memory_transfer()
    
    print("\n" + "=" * 50)
    print("比較完了")
    print("=" * 50)

# OpenCV GPU vs CPU パフォーマンス比較スクリプト (Webカメラ対応 + CuPy)

import cv2
import numpy as np
import cupy as cp
import time

print(f"使用ライブラリ: NumPy {np.__version__}, CuPy {cp.__version__}")

def process_cpu(frame):
    """CPU上で画像処理を実行（重い処理）"""
    # 4Kにリサイズ
    resized = cv2.resize(frame, (3840, 2160))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 複数のフィルタを適用（処理を増強）
    # 1. 大きなガウシアンブラー
    blurred1 = cv2.GaussianBlur(gray, (25, 25), 0)
    
    # 2. メディアンブラー
    median = cv2.medianBlur(blurred1, 9)
    
    # 3. 再度ガウシアンブラー
    blurred2 = cv2.GaussianBlur(median, (21, 21), 0)
    
    # 4. Cannyエッジ検出（しきい値を調整）
    edges = cv2.Canny(blurred2, 30, 100)
    
    # エッジを強調（見やすくする）
    edges = cv2.dilate(edges, None, iterations=1)
    
    # 元のサイズに戻す
    result = cv2.resize(edges, (frame.shape[1], frame.shape[0]))
    
    return result

def process_gpu(frame):
    """GPU上で画像処理を実行（CuPy + OpenCV CUDA使用）"""
    # フレームをCuPy配列に変換（GPUメモリ）
    frame_gpu = cp.asarray(frame)
    
    # OpenCV CUDA用にGpuMatに変換
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(cp.asnumpy(frame_gpu))
    
    # 4Kにリサイズ（OpenCV CUDA）
    gpu_resized = cv2.cuda.resize(gpu_frame, (3840, 2160))
    gpu_gray = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2GRAY)
    
    # 複数のフィルタを適用（処理を増強）
    # 1. 大きなガウシアンブラー
    gaussian_filter1 = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC1, cv2.CV_8UC1, (25, 25), 0
    )
    gpu_blurred1 = gaussian_filter1.apply(gpu_gray)
    
    # 2. メディアンブラー
    median_filter = cv2.cuda.createMedianFilter(cv2.CV_8UC1, 9)
    gpu_median = median_filter.apply(gpu_blurred1)
    
    # 3. 再度ガウシアンブラー
    gaussian_filter2 = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC1, cv2.CV_8UC1, (21, 21), 0
    )
    gpu_blurred2 = gaussian_filter2.apply(gpu_median)
    
    # 4. Cannyエッジ検出（しきい値を調整）
    canny_detector = cv2.cuda.createCannyEdgeDetector(30, 100)
    gpu_edges = canny_detector.detect(gpu_blurred2)
    
    # エッジを強調（見やすくする）
    morph_filter = cv2.cuda.createMorphologyFilter(
        cv2.MORPH_DILATE, cv2.CV_8UC1, np.ones((3, 3), np.uint8)
    )
    gpu_dilated = morph_filter.apply(gpu_edges)
    
    # 元のサイズに戻す
    gpu_result = cv2.cuda.resize(gpu_dilated, (frame.shape[1], frame.shape[0]))
    
    # 結果をCPUメモリにダウンロード
    edges = gpu_result.download()
    
    return edges

def main():
    print("="*60)
    print("   OpenCV GPU vs CPU パフォーマンス比較 (Webカメラ + CuPy)")
    print("="*60)
    
    # GPUデバイスの確認
    if not hasattr(cv2, 'cuda'):
        print("\n⚠ Error: OpenCVにCUDAモジュールがありません")
        print("CUDA対応版のOpenCVをインストールしてください。")
        return
    
    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        if gpu_count == 0:
            print("\n⚠ Error: CUDA対応のデバイスが見つかりません")
            return
        
        print(f"\n✓ CUDA対応デバイス数: {gpu_count}")
        device_id = cv2.cuda.getDevice()
        print(f"✓ 使用中のデバイス: {device_id}")
        
        # CuPyのデバイス情報
        cupy_device = cp.cuda.Device()
        print(f"✓ CuPyデバイス: {cupy_device.compute_capability}")
        print(f"✓ GPUメモリ: {cupy_device.mem_info[1] / 1024**3:.1f} GB")
        
        # デバイス情報を取得（OpenCV 4.xのAPI）
        try:
            device_info = cv2.cuda.DeviceInfo(device_id)
            print(f"✓ 計算能力: {device_info.majorVersion()}.{device_info.minorVersion()}")
        except:
            pass  # デバイス詳細情報が取得できない場合はスキップ
        
    except Exception as e:
        print(f"\n⚠ CUDA初期化エラー: {e}")
        return
    
    # Webカメラの初期化
    print("\nWebカメラを初期化しています...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("⚠ Webカメラが見つかりません。")
        print("利用可能なカメラデバイスを確認してください。")
        print("\n代替として、サンプル画像でデモを実行しますか? (y/n): ", end='')
        response = input().lower()
        if response != 'y':
            return
        use_camera = False
        print("\nサンプル画像モードで実行します...")
    else:
        use_camera = True
        # カメラの解像度を設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"✓ Webカメラを開きました: {int(actual_width)}x{int(actual_height)}")
    
    # サンプル画像の生成関数（カメラが使えない場合用）
    def generate_sample_frame():
        """テスト用のカラフルなサンプル画像を生成"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # カラフルなグラデーションとパターンを追加
        for i in range(0, 1280, 20):
            cv2.line(frame, (i, 0), (i, 720), (i % 255, (i*2) % 255, (i*3) % 255), 2)
        for i in range(0, 720, 20):
            cv2.line(frame, (0, i), (1280, i), ((i*3) % 255, (i*2) % 255, i % 255), 2)
        # 円とテキストを追加
        cv2.circle(frame, (640, 360), 200, (255, 255, 0), 5)
        cv2.putText(frame, 'GPU Computing Test', (400, 360), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        # ランダムノイズを追加
        noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        return frame
    
    # ウォームアップ
    print("\nGPUウォームアップ中...")
    if use_camera:
        ret, frame = cap.read()
        if not ret:
            frame = generate_sample_frame()
    else:
        frame = generate_sample_frame()
    
    try:
        _ = process_gpu(frame)
        print("✓ GPUウォームアップ完了")
    except Exception as e:
        print(f"⚠ GPU処理のウォームアップ中にエラー: {e}")
        import traceback
        traceback.print_exc()
        if use_camera:
            cap.release()
        return
    
    print("\n処理を開始します...")
    if use_camera:
        print("Webカメラから映像を取得中...")
    print("'q'キーで終了、ウィンドウを閉じても終了します\n")
    print(f"{'フレーム':<10} {'CPU (ms)':<12} {'GPU (ms)':<12} {'高速化':<10}")
    print("-" * 50)
    
    cpu_times = []
    gpu_times = []
    frame_count = 0
    max_frames = 100 if not use_camera else 0  # カメラモードは無制限
    
    try:
        while True:
            if use_camera:
                ret, frame = cap.read()
                if not ret:
                    print("\nエラー: カメラからフレームを取得できませんでした")
                    break
            else:
                frame = generate_sample_frame()
            
            frame_count += 1
            
            # CPU処理
            start_time = time.time()
            cpu_result = process_cpu(frame)
            cpu_time = (time.time() - start_time) * 1000
            cpu_times.append(cpu_time)
            
            # GPU処理
            start_time = time.time()
            gpu_result = process_gpu(frame)
            gpu_time = (time.time() - start_time) * 1000
            gpu_times.append(gpu_time)
            
            # 高速化率を計算
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            # 元のフレームをリサイズ（表示用）
            original_display = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
            
            # 結果を画像に描画
            cv2.putText(original_display, "Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(cpu_result, f"CPU: {cpu_time:.2f}ms", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(gpu_result, f"GPU: {gpu_time:.2f}ms", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(gpu_result, f"Speedup: {speedup:.2f}x", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # エッジ画像をカラーに変換（見やすくする）
            cpu_result_color = cv2.cvtColor(cpu_result, cv2.COLOR_GRAY2BGR)
            gpu_result_color = cv2.cvtColor(gpu_result, cv2.COLOR_GRAY2BGR)
            
            # 3つを横に並べて表示
            combined = np.hstack((original_display, cpu_result_color, gpu_result_color))
            window_name = 'CPU vs GPU Comparison (Webcam)' if use_camera else 'CPU vs GPU Comparison (Sample)'
            cv2.imshow(window_name, combined)
            
            # 10フレームごとに統計を表示
            if frame_count % 10 == 0:
                print(f"{frame_count:<10} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.2f}x")
            
            # サンプル画像モードの終了条件
            if not use_camera and max_frames > 0 and frame_count >= max_frames:
                print(f"\n{max_frames}フレームの処理が完了しました。")
                break
            
            # キー入力チェック
            wait_time = 1 if use_camera else 30
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'):
                print("\nユーザーによって中断されました")
                break
                
    except KeyboardInterrupt:
        print("\n\nキーボード割り込みで終了します")
    except Exception as e:
        print(f"\n処理中にエラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if use_camera:
            cap.release()
        cv2.destroyAllWindows()
    # CuPyで統計計算（GPU加速）
        cpu_times_gpu = cp.array(cpu_times)
        gpu_times_gpu = cp.array(gpu_times)
        speedup_array = cpu_times_gpu / gpu_times_gpu
        
        print("\n" + "="*60)
        print("   パフォーマンス統計 (CuPy使用)")
        print("="*60)
        print(f"処理フレーム数:     {len(cpu_times)}")
        print(f"\nCPU平均処理時間:   {float(cp.mean(cpu_times_gpu)):.2f} ms")
        print(f"CPU最小処理時間:   {float(cp.min(cpu_times_gpu)):.2f} ms")
        print(f"CPU最大処理時間:   {float(cp.max(cpu_times_gpu)):.2f} ms")
        print(f"CPU標準偏差:       {float(cp.std(cpu_times_gpu)):.2f} ms")
        print(f"\nGPU平均処理時間:   {float(cp.mean(gpu_times_gpu)):.2f} ms")
        print(f"GPU最小処理時間:   {float(cp.min(gpu_times_gpu)):.2f} ms")
        print(f"GPU最大処理時間:   {float(cp.max(gpu_times_gpu)):.2f} ms")
        print(f"GPU標準偏差:       {float(cp.std(gpu_times_gpu)):.2f} ms")
        print(f"\n平均高速化率:       {float(cp.mean(speedup_array)):.2f}x")
        print(f"最大高速化率:       {float(cp.max(speedup_array)):.2f}x")
        print(f"最小高速化率:       {float(cp.min(speedup_array)):.2f}x")
        print("="*60)

if __name__ == "__main__":
    main()
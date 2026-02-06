# tensorflowのCUDA対応テストコード
import tensorflow as tf
import time
import numpy as np

def check_gpu_availability():
    """GPU利用可能性を確認"""
    print("=" * 60)
    print("TensorFlow GPU利用可能性チェック")
    print("=" * 60)
    
    print(f"\nTensorFlowバージョン: {tf.__version__}")
    print(f"GPU利用可能: {tf.config.list_physical_devices('GPU')}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n検出されたGPU数: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            # GPU詳細情報を取得
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"    詳細: {details}")
            except:
                pass
    else:
        print("\nGPUが検出されませんでした")
    
    # CUDAとcuDNNのサポート確認
    print(f"\nCUDAサポート: {tf.test.is_built_with_cuda()}")
    print(f"GPU利用可能: {tf.test.is_gpu_available(cuda_only=False) if hasattr(tf.test, 'is_gpu_available') else 'N/A (TF 2.x)'}")

def compare_cpu_gpu_matrix_multiplication():
    """CPU vs GPU 行列乗算の比較"""
    print("\n" + "=" * 60)
    print("CPU vs GPU 行列乗算の比較")
    print("=" * 60)
    
    matrix_size = 5000
    print(f"\n行列サイズ: {matrix_size}x{matrix_size}")
    
    # CPU上での行列乗算
    print("\n[CPU]")
    with tf.device('/CPU:0'):
        cpu_a = tf.random.normal([matrix_size, matrix_size])
        cpu_b = tf.random.normal([matrix_size, matrix_size])
        
        start = time.time()
        cpu_result = tf.matmul(cpu_a, cpu_b)
        cpu_time = time.time() - start
        
        print(f"実行時間: {cpu_time:.4f}秒")
        print(f"結果の形状: {cpu_result.shape}")
        print(f"結果のサンプル:\n{cpu_result[:2, :2]}")
    
    # GPU上での行列乗算（利用可能な場合）
    if tf.config.list_physical_devices('GPU'):
        print("\n[GPU]")
        with tf.device('/GPU:0'):
            gpu_a = tf.random.normal([matrix_size, matrix_size])
            gpu_b = tf.random.normal([matrix_size, matrix_size])
            
            # ウォームアップ
            _ = tf.matmul(gpu_a, gpu_b)
            
            start = time.time()
            gpu_result = tf.matmul(gpu_a, gpu_b)
            gpu_time = time.time() - start
            
            print(f"実行時間: {gpu_time:.4f}秒")
            print(f"結果の形状: {gpu_result.shape}")
            print(f"結果のサンプル:\n{gpu_result[:2, :2]}")
            print(f"\n高速化率: {cpu_time / gpu_time:.2f}倍")
    else:
        print("\nGPUが利用できないため、GPU比較をスキップします")

def test_convolution_operations():
    """畳み込み演算のテスト（GPUで特に高速）"""
    print("\n" + "=" * 60)
    print("畳み込み演算のテスト")
    print("=" * 60)
    
    batch_size = 32
    image_size = 224
    channels = 3
    filters = 64
    
    print(f"\n入力: [{batch_size}, {image_size}, {image_size}, {channels}]")
    print(f"フィルター数: {filters}")
    
    # ダミーの画像データを作成
    input_data = tf.random.normal([batch_size, image_size, image_size, channels])
    
    # CPU上での畳み込み
    print("\n[CPU]")
    with tf.device('/CPU:0'):
        conv_layer = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        
        start = time.time()
        cpu_result = conv_layer(input_data)
        cpu_time = time.time() - start
        
        print(f"実行時間: {cpu_time:.4f}秒")
        print(f"出力形状: {cpu_result.shape}")
    
    # GPU上での畳み込み
    if tf.config.list_physical_devices('GPU'):
        print("\n[GPU]")
        with tf.device('/GPU:0'):
            conv_layer_gpu = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
            
            # ウォームアップ
            _ = conv_layer_gpu(input_data)
            
            start = time.time()
            gpu_result = conv_layer_gpu(input_data)
            gpu_time = time.time() - start
            
            print(f"実行時間: {gpu_time:.4f}秒")
            print(f"出力形状: {gpu_result.shape}")
            print(f"\n高速化率: {cpu_time / gpu_time:.2f}倍")
    else:
        print("\nGPUが利用できないため、GPU比較をスキップします")

def test_training_step():
    """簡単な学習ステップのテスト"""
    print("\n" + "=" * 60)
    print("ニューラルネットワーク学習ステップのテスト")
    print("=" * 60)
    
    # 簡単なモデルを作成
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # ダミーデータ
    batch_size = 128
    num_batches = 100
    x_train = tf.random.normal([batch_size * num_batches, 784])
    y_train = tf.random.uniform([batch_size * num_batches], maxval=10, dtype=tf.int32)
    
    print(f"\nデータサイズ: {x_train.shape}")
    print(f"バッチサイズ: {batch_size}")
    print(f"バッチ数: {num_batches}")
    
    # GPU利用可能かどうかで表示を変更
    if tf.config.list_physical_devices('GPU'):
        print("\n使用デバイス: GPU")
    else:
        print("\n使用デバイス: CPU")
    
    # 学習実行
    start = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=3,
        verbose=0
    )
    training_time = time.time() - start
    
    print(f"\n学習時間: {training_time:.4f}秒")
    print(f"最終損失: {history.history['loss'][-1]:.4f}")
    print(f"最終精度: {history.history['accuracy'][-1]:.4f}")

def check_memory_usage():
    """GPUメモリ使用状況の確認"""
    print("\n" + "=" * 60)
    print("GPUメモリ使用状況")
    print("=" * 60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # メモリ増加を制限
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print("\nメモリ増加モード: 有効")
            print("（必要に応じてメモリを動的に割り当て）")
        except RuntimeError as e:
            print(f"\nメモリ設定エラー: {e}")
    else:
        print("\nGPUが検出されませんでした")

if __name__ == "__main__":
    print("\nTensorFlow CUDA対応テスト")
    print("=" * 60)
    
    # GPU利用可能性チェック
    check_gpu_availability()
    
    # メモリ設定
    check_memory_usage()
    
    # 各種テスト実行
    compare_cpu_gpu_matrix_multiplication()
    test_convolution_operations()
    test_training_step()
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
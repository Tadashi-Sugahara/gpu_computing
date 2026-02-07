#!/bin/bash
# OpenCV CUDA対応版のインストールスクリプト

set -e

echo "=== OpenCV CUDA対応版インストール ==="
echo ""
echo "GPU: NVIDIA GeForce RTX 2060"
echo "CUDA: 12.0"
echo ""

# 必要なパッケージをインストール
echo "ステップ 1: 必要なパッケージをインストール"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    gcc-12 \
    g++-12 \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-dev \
    python3-numpy \
    libtbb-dev \
    libdc1394-dev

echo ""
echo "ステップ 2: OpenCVソースをダウンロード"
mkdir -p ~/Downloads
cd ~/Downloads
mkdir -p opencv_build && cd opencv_build

if [ ! -d "opencv" ]; then
    git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git
fi

if [ ! -d "opencv_contrib" ]; then
    git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv_contrib.git
fi

echo ""
echo "ステップ 3: ビルド設定（CUDA有効）"
cd opencv
mkdir -p build && cd build

# CUDA 12.0と互換性のあるGCC-12を使用
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D CMAKE_C_COMPILER=/usr/bin/gcc-12 \
    -D CMAKE_CXX_COMPILER=/usr/bin/g++-12 \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=OFF \
    -D OPENCV_DNN_CUDA=OFF \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D CUDA_ARCH_BIN=7.5 \
    -D WITH_CUBLAS=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D HAVE_opencv_python3=ON \
    -D PYTHON_EXECUTABLE=$(which python3) \
    -D BUILD_EXAMPLES=ON ..

echo ""
echo "ステップ 4: ビルド（これには30-60分かかります）"
echo "使用可能なCPUコア数: $(nproc)"
# メモリ不足を避けるため、並列ジョブ数を制限
JOBS=$(($(nproc) / 2))
if [ $JOBS -lt 1 ]; then
    JOBS=1
fi
echo "並列ジョブ数: $JOBS"
make -j$JOBS

echo ""
echo "ステップ 5: インストール"
sudo make install
sudo ldconfig

echo ""
echo "=== インストール完了 ==="
echo "以下のコマンドで確認してください："
echo "python3 -c \"import cv2; print('OpenCV Version:', cv2.__version__); print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())\""

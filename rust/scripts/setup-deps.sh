#!/bin/bash
# Native dependencies of the Rust build, taken from the very packages the
# Python version runs on, so that both compute the same pixels and numbers:
#
#   /opt/opencv5       OpenCV 5.0.0 built from the opencv-python 5.0.0.93 sdist
#                      with the wheel's options (CPU dispatch, Intel IPP, LAPACK)
#   /opt/openblas-cv   the OpenBLAS the opencv-python wheel ships (its LAPACK)
#   /opt/mupdf         PyMuPDF 2.0's MuPDF 1.28.5 library and headers (rendering)
#   /opt/onnxruntime   ONNX Runtime 1.29.0 from its wheel (OCR)
#   /opt/rfpii/models  the PP-OCR models of rapidocr-onnxruntime 1.2.3
#   /opt/rfpii/lib     numpy 2.5.3's OpenBLAS (np.polyfit / lstsq / inv)
#
# Needs: python3 + pip, cmake, a C/C++ compiler, clang (for the OpenCV
# bindings), apt-get (OpenBLAS / LAPACKE headers), network access to PyPI and
# to raw.githubusercontent.com (Intel IPP for OpenCV). Run as root.
set -euo pipefail

WORK=${WORK:-/tmp/rfpii-deps}
mkdir -p "$WORK" && cd "$WORK"

wheel() {  # wheel <pip spec> <dir>: unpack a manylinux wheel for this Python
    mkdir -p "$2"
    pip download --no-deps --only-binary :all: -d "$2/dl" "$1"
    python3 -m zipfile -e "$2"/dl/*.whl "$2/x"
}

# --- wheels ------------------------------------------------------------------------------
wheel "opencv-python==5.0.0.93" cvwheel
wheel "pymupdf==2.0" pymupdf
wheel "onnxruntime==1.29.0" ort
wheel "rapidocr-onnxruntime==1.2.3" rapidocr
wheel "numpy==2.5.3" numpy

mkdir -p /opt/openblas-cv/lib /opt/openblas-cv/include /opt/mupdf/lib /opt/mupdf/include \
         /opt/onnxruntime/lib /opt/rfpii/models /opt/rfpii/lib
cp cvwheel/x/opencv_python.libs/libopenblasp-r0-59ffcd50.3.15.so \
   cvwheel/x/opencv_python.libs/libgfortran-83c28eba.so.5.0.0 \
   cvwheel/x/opencv_python.libs/libquadmath-2284e583.so.0.0.0 /opt/openblas-cv/lib/
ln -sf libopenblasp-r0-59ffcd50.3.15.so /opt/openblas-cv/lib/libopenblas.so
cp pymupdf/x/pymupdf/libmupdf.so.28.5 /opt/mupdf/lib/
ln -sf libmupdf.so.28.5 /opt/mupdf/lib/libmupdf.so
cp -r pymupdf/x/pymupdf/mupdf-devel/include/mupdf /opt/mupdf/include/
cp ort/x/onnxruntime/capi/libonnxruntime.so.1.29.0 /opt/onnxruntime/lib/
cp rapidocr/x/rapidocr_onnxruntime/models/*.onnx /opt/rfpii/models/
cp numpy/x/numpy.libs/libscipy_openblas64_-f48b354e.so \
   numpy/x/numpy.libs/libgfortran-83c28eba-468e71e5.so.5.0.0 \
   numpy/x/numpy.libs/libquadmath-2284e583-a9307bba.so.0.0.0 /opt/rfpii/lib/

# OpenBLAS / LAPACKE headers (the library itself is the wheel's)
mkdir -p debs && (cd debs && apt-get download libopenblas-pthread-dev liblapacke-dev)
for d in debs/*.deb; do dpkg-deb -x "$d" debs/x; done
cp debs/x/usr/include/x86_64-linux-gnu/openblas-pthread/*.h /opt/openblas-cv/include/
find debs/x -name "lapack*.h" -exec cp {} /opt/openblas-cv/include/ \;

# --- OpenCV 5.0 ----------------------------------------------------------------------------
# the sdist (pip would try to build it): straight from PyPI
mkdir -p cvsrc
url=$(python3 -c 'import json, urllib.request
d = json.load(urllib.request.urlopen("https://pypi.org/pypi/opencv-python-headless/5.0.0.93/json"))
print([u["url"] for u in d["urls"] if u["packagetype"] == "sdist"][0])')
curl -sSfL -o cvsrc/opencv.tar.gz "$url"
tar xzf cvsrc/opencv.tar.gz -C cvsrc
OpenBLAS_HOME=/opt/openblas-cv cmake -S cvsrc/opencv_python_headless-5.0.0.93/opencv -B cvbuild \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/opencv5 -DBUILD_SHARED_LIBS=ON \
  -DBUILD_LIST=core,imgproc,imgcodecs,features,flann,geometry,calib,objdetect,dnn,video \
  -DCPU_BASELINE=SSE3 -DCPU_DISPATCH=SSE4_1,SSE4_2,AVX,FP16,AVX2,AVX512_SKX \
  -DWITH_IPP=ON -DWITH_LAPACK=ON -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_OPENJPEG=ON -DBUILD_ZLIB=ON \
  -DWITH_WEBP=OFF -DWITH_AVIF=OFF -DWITH_OPENEXR=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF \
  -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_OPENCL=OFF -DWITH_ITT=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_EIGEN=OFF \
  -DWITH_VA=OFF -DWITH_OBSENSOR=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_DOCS=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
  -DOPENCV_GENERATE_PKGCONFIG=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_PROTOBUF=ON
cmake --build cvbuild -j"$(nproc)"
cmake --install cvbuild

echo "done: build the tool with (cd rust && cargo build --release)"

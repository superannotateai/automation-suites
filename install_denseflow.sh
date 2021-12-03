# fetch install scripts
git clone https://github.com/innerlee/setup.git
cd setup

# opencv depends on ffmpeg for video decoding
# ffmpeg depends on nasm, yasm, libx264, libx265, libvpx
./zznasm.sh
./zzyasm.sh
./zzlibx264.sh
./zzlibx265.sh
./zzlibvpx.sh
# finally install ffmpeg
./zzffmpeg.sh

# install opencv 4.3.0
./zzopencv.sh

# install boost
./zzboost.sh

# install hdf5
./zzhdf5.sh

# finally, install denseflow
./zzdenseflow.sh
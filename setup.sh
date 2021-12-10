#The setup script assumes that you have correct cuda version installed on your machine (10.x)
#It is preferable to create a separate virtual environment before running this script

pip install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0
pip install superannotate
pip install mmcv
pip install youtube-dl

git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -r requirements/build.txt
pip install -v -e .
cd ..
git clone -b temporal_action_localization https://github.com/superannotateai/automation-suites.git

export ZZROOT=/root/app
export PATH="$ZZROOT"/bin:$PATH
export LD_LIBRARY_PATH="$ZZROOT"/lib:"$ZZROOT"/lib64:$LD_LIBRARY_PATH
export OpenCV_DIR=$ZZROOT
export BOOST_ROOT=$ZZROOT

chmod a+x automation-suites/install_densflow.sh
sh automation-suites/install_densflow.sh
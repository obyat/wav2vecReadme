<h1 align="center">Welcome to WavBERT: Exploiting Semantic and Non-semantic Speech using Wav2vec and BERT for Dementia Detection 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="http://www.homepages.ed.ac.uk/sluzfil/ADReSSo-2021/" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="https://github.com/kefranabg/readme-md-generator/graphs/commit-activity" target="_blank">
    <img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" />
  </a>
</p>

> In this project, we exploit semantic and non-semantic information from patient’s speech data usingWav2vec and Bidirectional Encoder Representations from Transformers (BERT) for dementia detection. We first propose a basic WavBERT model by extracting semantic information from speech data using Wav2vec, and analyzing the semantic information using BERT for dementia detection. While the basic model discards the non-semantic information, we propose extended WavBERT models that convert the output ofWav2vec to the input to BERT for preserving the non-semantic information in dementia detection. Specifically, we determine the locations and lengths of inter-word pauses using the number of blank tokens from Wav2vec where the threshold for setting the pauses is automatically generated via BERT. We further design a pre-trained embedding conversion network that converts the output embedding of Wav2vec to the input embedding of BERT, enabling the fine-tuning of WavBERT with non-semantic information. Our evaluation results using the ADReSSo dataset showed that the WavBERT models achieved the highest accuracy of 83.1% in the classification task, the lowest Root-Mean-Square Error (RMSE) score of 4.44 in the regression task, and a mean F1 of 70.91% in the progression task. We confirmed the effectiveness of WavBERT models exploiting both semantic and non-semantic speech.

### 🏠 [Homepage](https://github.com/billzyx/wav2vec)

## Run tests

```sh
.....
```
## Author

👤 **Xiaohui Liang**

* Website: http://faculty.umb.edu/xiaohui.liang/
* Website: https://www.linkedin.com/in/xiaohui-liang-7622a419/
## Author

👤 **Youxiang Zhu**

* Website: http://www.faculty.umb.edu/xiaohui.liang/mobcp.html
* GitHub: [@ billzyx ](https://github.com/billzyx )

## Author

👤 **Abdelrahman Obyat**

* Website: https://www.linkedin.com/in/abdelrahman-obyat-52065b173/
* GitHub: [@ obyat ](https://github.com/obyat)

# Installation
### Python3.5 or newer is required for this project

### Installing python3.8:
```
sudo apt update  
sudo apt install software-properties-common  
sudo add-apt-repository ppa:deadsnakes/ppa  
sudo apt install python3.8  
sudo apt install python3-pip  
```


## Installing env:
```
sudo apt-get install python3.8-venv  
```


Please note, if root system is full and no system storage available error is encountered. The following is recommended:
```
pip3 cache purge
```
Restarting system in order also clear root cache
s { text-decoration:none; } //strike-through
em { font-style: normal; font-weight: bold; } //italic emphasis


// colors
s { color: green }
em { color: blue }
<span style="color:red">some **This is Red Bold.** text</span>
## Please note, the following must be install in env variable using the following commands:
1. python3.8 -m venv env
2. source env/bin/activate

# Installing Torch:

python3.8 -m pip --no-cache-dir install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
Looking in links: https://download.pytorch.org/whl/torch_stable.html

Customization of Pytorch for different builds can be found here:
https://pytorch.org/get-started/locally/

Please note, if pip wheel errors are encountered, reinstalling pip wheel is recommended:
1. pip install wheel


# Installing Additional Libraries:

1. python3.8 -m pip install transformers

2. python3.8 -m pip install tqdm
3. python3.8 -m pip install numpy

4. python3.8 -m pip install matplotlib
5. python3.8 -m pip install jiwer

6. python3.8 -m pip install librosa
7. python3.8 -m pip install fairseq

8. python3.8 -m pip install datasets

9. python3.8 -m pip --no-cache-dir install editdistance
10. python3.8 -m pip --no-cache-dir install sentencepiece

## Wav2Vec
1. Follow here: https://medium.com/@shaheenkader/how-to-install-wav2letter-dc94c3b74e97

# Installing Arrayfire:
1. sudo apt-get install cmake g++
2. wget https://arrayfire.s3.amazonaws.com/3.6.1/ArrayFire-no-gl-v3.6.1_Linux_x86_64.sh
3. chmod u+x ArrayFire-no-gl-v3.6.1_Linux_x86_64.sh
4. sudo bash ArrayFire-no-gl-v3.6.1_Linux_x86_64.sh --include-subdir --prefix=/opt
5. sudo bash -c 'echo /opt/arrayfire-no-gl/lib > /etc/ld.so.conf.d/arrayfire.conf'
6. sudo ldconfig 

# Installing GoogleTest:
1. sudo apt-get install libgtest-dev
2. sudo apt-get install cmake # install cmake
3. cd /usr/src/gtest
4. sudo cmake CMakeLists.txt
5. sudo make
6. cd lib
7. sudo cp *.a /usr/lib

# Installing OpenMPI:
1. sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

[4] flashlight:

# Installing CUDA:
1. wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
2. sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
3. sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
4. sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
*before falshlight
5. sudo apt-get update
6. sudo apt-get -y install cuda
7. sudo apt install nvidia-cuda-toolkit
8. pip3.8 install mkl-devel



//websites:
https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/

https://arrayfire.org/docs/installing.htm

https://askubuntu.com/questions/103643/cannot-echo-hello-x-txt-even-with-sudo

https://gitlab.kitware.com/cmake/cmake/-/issues/19396


## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_

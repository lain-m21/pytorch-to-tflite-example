# pytorch-to-tflite-example
Convert MobileNetV3Small defined and pre-trained in PyTorch to a TFLite quantized model

## Requirements

Python >= 3.6.0

Python packages:

- `Keras==2.2.4`
- `onnx==1.5.0`
- `onnx2keras==0.0.3`
- `tensorflow==1.14.0`
- `torch==1.1.0`
- `Pillow==6.1.0`

## Usage

### Download weights
Download manually from [here](https://github.com/d-li14/mobilenetv3.pytorch/raw/master/pretrained/mobilenetv3-large-657e7b3d.pth)  
or use  

    ./download_weight.sh

### Run the script

    python3 main.py

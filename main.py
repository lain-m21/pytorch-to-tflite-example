from pathlib import Path
import torch
from mobilenetv3 import mobilenetv3_large
from converters import pytorch2savedmodel, savedmodel2tflite


def main():
    data_dir = Path.cwd().joinpath('data')
    data_dir.mkdir(exist_ok=True)

    model_torch = mobilenetv3_large()
    state_dict = torch.load('pretrained/mobilenetv3-large-657e7b3d.pth', map_location='cpu')
    model_torch.clean_and_load_state_dict(state_dict)
    model_torch.convert_se()

    onnx_model_path = str(data_dir.joinpath('model.onnx'))
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ['image_array']
    output_names = ['category']

    torch.onnx.export(model_torch, dummy_input, onnx_model_path,
                      input_names=input_names, output_names=output_names)

    saved_model_dir = str(data_dir.joinpath('saved_model'))
    pytorch2savedmodel(onnx_model_path, saved_model_dir)

    tflite_model_path = str(data_dir.joinpath('model.tflite'))
    savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=False)

    tflite_quantized_model_path = str(data_dir.joinpath('model_quantized.tflite'))
    savedmodel2tflite(saved_model_dir, tflite_quantized_model_path, quantize=True)


if __name__ == '__main__':
    main()

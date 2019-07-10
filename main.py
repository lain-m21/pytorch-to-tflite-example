import logging
from pathlib import Path
import numpy as np
import torch
from mobilenetv3 import mobilenetv3_large
from converters import pytorch2savedmodel, savedmodel2tflite
from image import load_and_preprocess_image
from tflite import get_tflite_outputs


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    logger.info('Create data directory in which models dumped.\n')
    data_dir = Path.cwd().joinpath('data')
    data_dir.mkdir(exist_ok=True)

    logger.info('\nInitialize MobileNetV3 and load pre-trained weights\n')
    model_torch = mobilenetv3_large()
    state_dict = torch.load('pretrained/mobilenetv3-large-657e7b3d.pth', map_location='cpu')
    model_torch.clean_and_load_state_dict(state_dict)

    logger.info('\nConvert Squeeze and Excitation modules to convert the model to a Keras model.\n')
    model_torch.convert_se()

    for m in model_torch.modules():
        m.training = False

    onnx_model_path = str(data_dir.joinpath('model.onnx'))
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ['image_array']
    output_names = ['category']

    logger.info(f'\nExport PyTorch model in ONNX format to {onnx_model_path}.\n')
    torch.onnx.export(model_torch, dummy_input, onnx_model_path,
                      input_names=input_names, output_names=output_names)

    saved_model_dir = str(data_dir.joinpath('saved_model'))
    logger.info(f'\nConvert ONNX model to Keras and save as saved_model.pb.\n')
    pytorch2savedmodel(onnx_model_path, saved_model_dir)

    logger.info(f'\nConvert saved_model.pb to TFLite model.\n')
    tflite_model_path = str(data_dir.joinpath('model.tflite'))
    tflite_model = savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=False)

    logger.info(f'\nConvert saved_model.pb to TFLite quantized model.\n')
    tflite_quantized_model_path = str(data_dir.joinpath('model_quantized.tflite'))
    tflite_quantized_model = savedmodel2tflite(saved_model_dir, tflite_quantized_model_path, quantize=True)

    logger.info("\nCompare PyTorch model's outputs and TFLite models' outputs.\n")
    num_same_outputs = 0
    image_path_list = list(Path('tools').glob('*.jpg'))
    for path in image_path_list:
        input_array = load_and_preprocess_image(str(path))
        input_tensor = torch.from_numpy(input_array)

        torch_output = model_torch(input_tensor).data.numpy().reshape(-1, )
        tflite_output = get_tflite_outputs(input_array.transpose((0, 2, 3, 1)), tflite_model).reshape(-1, )
        logger.info(f'PyTorch - first 5 items: {torch_output[:5]}')
        logger.info(f'TFLite - first 5 items: {tflite_output[:5]}\n')

        torch_output_index = np.argmax(torch_output)
        tflite_output_index = np.argmax(tflite_output)

        if torch_output_index == tflite_output_index:
            num_same_outputs += 1

    logger.info(f'# of matched outputs: {num_same_outputs} / {len(image_path_list)}\n')

    logger.info("\nCompare PyTorch model's outputs and TFLite quantized models' outputs.\n")
    num_same_outputs = 0
    image_path_list = list(Path('tools').glob('*.jpg'))
    for path in image_path_list:
        input_array = load_and_preprocess_image(str(path))
        input_tensor = torch.from_numpy(input_array)

        torch_output = model_torch(input_tensor).data.numpy().reshape(-1, )
        tflite_output = get_tflite_outputs(input_array.transpose((0, 2, 3, 1)), tflite_quantized_model).reshape(-1, )
        logger.info(f'PyTorch - first 5 items: {torch_output[:5]}')
        logger.info(f'TFLite - first 5 items: {tflite_output[:5]}\n')

        torch_output_index = np.argmax(torch_output)
        tflite_output_index = np.argmax(tflite_output)

        if torch_output_index == tflite_output_index:
            num_same_outputs += 1

    logger.info(f'# of matched outputs: {num_same_outputs} / {len(image_path_list)}\n')


if __name__ == '__main__':
    main()

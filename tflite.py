import tensorflow as tf


def get_tflite_outputs(input_array, tflite_model):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()

    tflite_results = interpreter.get_tensor(output_details[0]['index'])
    return tflite_results

from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="model_fp32.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QInt8
)

print("ONNX model quantized and saved as model_int8.onnx.")
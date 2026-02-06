#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
	Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "SpleenSegmenter");
	Ort::SessionOptions session_options;


	OrtCUDAProviderOptions cuda_options{};
	session_options.AppendExecutionProvider_CUDA(cuda_options);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	const char* model_path = "outputs/model_spleen.onnx";
	std::cout << "Loading model from " << model_path << std::endl;

	try {
		Ort::Session session(env, model_path, session_options);
		std::cout << "Model loaded successfully in C++" << std::endl;

		// Input shape: (Batch=1, Channel=1, D=96, H=96, W=96)
		std::vector<int64_t> input_shape = {1, 1, 96, 96, 96};
		size_t input_tensor_size = 96 * 96 * 96;

		// Initialize input tensor with random values
		std::vector<float> input_tensor_values(input_tensor_size);
		for (size_t i = 0; i < input_tensor_size; i++)
			input_tensor_values[i] = 0.5f; // fill with 0.5 to simulate a typical CT scan

		Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
			OrtArenaAllocator, OrtMemTypeDefault);

		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
			memory_info,
			input_tensor_values.data(),
			input_tensor_size,
			input_shape.data(),
			input_shape.size()
		);

		const char* input_names[] = {"input"};
		const char* output_names[] = {"output"};

		auto output_tensors = session.Run(
			Ort::RunOptions{nullptr},
			input_names,
			&input_tensor,
			1,
			output_names,
			1
		);

		float* floatarray = output_tensors[0].GetTensorMutableData<float>();
		std::cout << "Inference result: " << floatarray[0] << std::endl;

		return 0;
	} catch (const Ort::Exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
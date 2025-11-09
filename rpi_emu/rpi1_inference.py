import numpy as np
from PIL import Image
import pathlib
from ai_edge_litert.interpreter import Interpreter
from sklearn.metrics import classification_report, confusion_matrix
import time
import os

MODEL_PATH  = "mobilenetv2_int8_ptq.tflite"
DATASET_DIR = "train/images"
RESULT_FILE = "results.txt"


def test_latency(litert_interpreter: Interpreter, num_runs=100) -> str:

    input_details = litert_interpreter.get_input_details()
    input_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape']

    output_str = ""

    if input_dtype == np.uint8:
        # Full INT8 models expect UINT8 input (0-255 range)
        output_str += "-> Testing FULL INT8 PTQ MODEL (Input Type: UINT8)"
        input_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    else:
        # FP32/Dynamic models expect FLOAT32 input (e.g., -1 to 1 range)
        output_str += "-> Testing DYNAMIC PTQ MODEL (Input Type: FLOAT32)"
        input_data = np.random.uniform(-1, 1, size=input_shape).astype(np.float32)

    litert_interpreter.set_tensor(input_details[0]['index'], input_data)
    # Warm-up
    litert_interpreter.invoke()

    start_time = time.time()
    for _ in range(num_runs):
        litert_interpreter.invoke()

    end_time = time.time()

    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
    output_str += "\nAverage Latency (ms): " + str(avg_latency_ms) + "\n"

    return output_str

def evaluate_full_integer_model(litert_interpreter: Interpreter, val_ds_path: str) -> str:
    """
    Evaluates a full-integer TFLite model that requires int8 input.
    """
    output_str = ""

    input_details = litert_interpreter.get_input_details()[0]
    output_details = litert_interpreter.get_output_details()[0]

    # Get quantization parameters
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    _, img_height, img_width, _ = input_details['shape']
    data_dir = pathlib.Path(val_ds_path)
    class_names = sorted([d.name for d in data_dir.glob('*') if d.is_dir()])

    if not class_names:
        output_str += f"Error: No subdirectories found in {val_ds_path}."
        return output_str

    image_paths = []
    true_labels = []

    for i, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        for img_path in list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png')):
            image_paths.append(str(img_path))
            true_labels.append(i)

    if len(image_paths) == 0:
        output_str += f"\nError: No images found in class subdirectories.\n"
        return output_str

    output_str += f"\nFound {len(image_paths)} images in {len(class_names)} classes.\n"

    predicted_labels = []
    processed_true_labels = []

    for i, image_path in enumerate(image_paths):
        try:
            # Preprocess Image (to float32)
            img = Image.open(image_path).convert('L').resize((img_width, img_height))
            image_float = np.array(img).astype(np.float32)
            image_float_expanded = np.expand_dims(image_float, axis=-1)
            image_float_expanded = np.expand_dims(image_float_expanded, axis=0)

            # (float32 / scale) + zero_point = int8
            image_quantized = (image_float_expanded / input_scale) + input_zero_point
            image_quantized = image_quantized.astype(input_details['dtype'])

            # Inference
            litert_interpreter.set_tensor(input_details['index'], image_quantized)
            litert_interpreter.invoke()

            # De-quantize Output
            output_quantized = litert_interpreter.get_tensor(output_details['index'])
            # (int8 - zero_point) * scale = float32
            output_float = (output_quantized.astype(np.float32) - output_zero_point) * output_scale

            predicted_label = np.argmax(output_float[0])
            predicted_labels.append(predicted_label)
            processed_true_labels.append(true_labels[i])

        except Exception as e:
            output_str += f"\nWarning: Skipping {image_path}, failed to process. Error: {e}\n"
            continue

    if not predicted_labels:
        output_str += "\nNo inferences were successfully run.\n"
        return output_str

    true_labels_np = np.array(processed_true_labels)
    predicted_labels_np = np.array(predicted_labels)

    output_str += "\n--- Classification Report ---"
    output_str += "\n" + str(classification_report(
        true_labels_np, predicted_labels_np, target_names=class_names, zero_division=0
    )) + "\n"

    output_str += "\n--- Confusion Matrix ---"
    output_str += "\n" + str(confusion_matrix(true_labels_np, predicted_labels_np)) + "\n"

    accuracy = np.mean(predicted_labels_np == true_labels_np)

    output_str += f"\nOverall Accuracy: {accuracy * 100:.2f}%\n"

    return output_str

interpreter = Interpreter(MODEL_PATH)
interpreter.allocate_tensors()

with open(RESULT_FILE, 'w') as f:
    f.write(str(evaluate_full_integer_model(interpreter, DATASET_DIR)))
    f.write(f"Average inference time: {test_latency(interpreter, num_runs=100)}")

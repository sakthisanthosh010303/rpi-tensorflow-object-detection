# Author: Sakthi Santhosh
# Created on: 08/09/2022
#
# Tensorflow Object Detection - Custom Model
def set_input_tensor(tflite_handle, input_details, image) -> None:
    from numpy import expand_dims

    input_tensor = tflite_handle.tensor(input_details[0]["index"])()[0]
    input_tensor[:, :] = expand_dims((image - 255) / 255, axis=0)

def main(argv) -> int:
    if len(argv) < 1:
        print("Error: Program called with no data.")
        return 1

    from os import path

    if not path.exists(argv[0]):
        print("Error: Invalid path.")
        return 1

    from cv2 import (
        COLOR_BGR2RGB,
        cvtColor,
        imread,
        imwrite,
        rectangle,
        resize
    )
    from numpy import squeeze
    from tflite_runtime.interpreter import Interpreter

    DETECTION_THRESHOLD = 0.8
    MODEL = "./model/custom.tflite"

    # Create label map.
    label_map = {}
    with open("./label/custom.txt", 'r') as file_handle:
        for index, object in enumerate(file_handle.read().strip().split(',')):
            label_map[index] = object

    # Set-up Tensorflow runtime.
    tflite_handle = Interpreter(MODEL)
    tflite_handle.allocate_tensors()
    input_details = tflite_handle.get_input_details()
    output_details = tflite_handle.get_output_details()

    # Process image for detection.
    image_handle = imread(argv[0])
    image_dimensions = image_handle.shape
    image_processed = resize(
        cvtColor(image_handle, COLOR_BGR2RGB),
        (input_details[0]["shape"][2], input_details[0]["shape"][1])
    )

    # Detect from images.
    set_input_tensor(tflite_handle, input_details, image_processed)
    tflite_handle.invoke()

    scores, boxes, count, classes = (
        squeeze(tflite_handle.get_tensor(output_details[0]["index"])),
        squeeze(tflite_handle.get_tensor(output_details[1]["index"])),
        int(squeeze(tflite_handle.get_tensor(output_details[2]["index"]))),
        squeeze(tflite_handle.get_tensor(output_details[3]["index"])),
    )
    results = []

    # Process detected data.
    for counter in range(count):
        if scores[counter] > DETECTION_THRESHOLD:
            results.append({
                "bounding_box": boxes[counter],
                "class_id": classes[counter],
                "score": scores[counter]
            })

    if not results:
        print("Error: No object detected.")
        return 1

    # Draw contour on image.
    print("Image dimensions:", image_dimensions)
    for result in results:
        ymin, xmin, ymax, xmax = result["bounding_box"]
        xmin = int(max(1,xmin * image_dimensions[1]))
        xmax = int(min(image_dimensions[1], xmax * image_dimensions[1]))
        ymin = int(max(1, ymin * image_dimensions[0]))
        ymax = int(min(image_dimensions[0], ymax * image_dimensions[0]))

        print("\nResult:", result)
        print("Contour: (%d, %d), (%d, %d)"%(xmin, ymin, xmax, ymax))
        rectangle(image_handle, (xmin, ymin), (xmax, ymax), (255, 255, 255), 5)

    imwrite("./workspace/output/temp_image.jpg", image_handle)
    print("\nOutput: ./workspace/output/temp_image.jpg")
    return 0

if __name__ == "__main__":
    from sys import argv

    exit(main(argv[1:]))

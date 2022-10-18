# Author: Sakthi Santhosh
# Created on: 08/09/2022
#
# Tensorflow Object Detection - SSD MobileNet
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
        FONT_HERSHEY_SIMPLEX,
        imread,
        imwrite,
        putText,
        rectangle,
        resize
    )
    from numpy import (
        expand_dims,
        squeeze
    )
    from tflite_runtime.interpreter import Interpreter

    DETECTION_THRESHOLD = 0.5
    MODEL = "./model/ssd_mobilenet.tflite"

    # Create label map.
    label_map = {}
    with open("./label/ssd_mobilenet.txt", 'r') as file_handle:
        for index, object in enumerate(file_handle.readlines()):
            label_map[index] = object.strip()

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
    tflite_handle.set_tensor(input_details[0]["index"], expand_dims(image_processed, axis=0))
    tflite_handle.invoke()

    scores, boxes, classes = (
        tflite_handle.get_tensor(output_details[2]["index"])[0],
        tflite_handle.get_tensor(output_details[0]["index"])[0],
        tflite_handle.get_tensor(output_details[1]["index"])[0],
    )
    results = []

    # Process detected data.
    for counter in range(len(scores)):
        if scores[counter] > DETECTION_THRESHOLD and scores[counter] <= 1.0:
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
        putText(
            image_handle,
            "%s %f%%"%(label_map[int(result["class_id"])], result["score"] * 100),
            (xmin, ymin - 10),
            FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

    imwrite("./workspace/output/temp_image.jpg", image_handle)
    print("\nOutput: ./workspace/output/temp_image.jpg")
    return 0

if __name__ == "__main__":
    from sys import argv

    exit(main(argv[1:]))

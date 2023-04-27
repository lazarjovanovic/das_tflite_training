from datetime import datetime
from tflite_model_maker import image_classifier
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2


# classes = ['???'] * model.model_spec.config.num_classes
# COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)
classes = ["bcc", "psoriasis"]
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)


def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    global classes
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    output_class_index = output.argmax()
    output_class = classes[output_class_index]
    output_probability = round(output[output_class_index] / 256, 5) * 100

    signature_fn = interpreter.get_signature_runner()

    # Feed the input image to the model
    output = signature_fn(images=image)

    # Get all outputs from the model
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # Run object detection on the input image
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # Find the class index of the current object
        class_id = int(obj['class_id'])

        # Draw the bounding box and label on the image
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        # Make adjustments to make the label visible for all objects
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
        cv2.putText(original_image_np, label, (xmin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Return the final image
    original_uint8 = original_image_np.astype(np.uint8)
    return original_uint8


if __name__ == '__main__':
    path_model = 'model.tflite'
    # path_image = 'data/bcc_image_50.jpeg'
    path_image = 'data/psoriasis_image_46.jpeg'
    DETECTION_THRESHOLD = 0.3

    # labels = ['bcc', 'psoriasis']
    #
    # # model = TensorflowLiteClassificationModel(path_model, labels, image_size=416)
    # model = TensorflowLiteClassificationModel(path_model, labels)
    # # (label, probability) = model.run_from_filepath(path_image)
    # print(datetime.now())
    # label_probability = model.run_from_filepath(path_image)
    # print(datetime.now())

    # WORKING INSTANCE
    main_interpreter = tf.lite.Interpreter(model_path=path_model)
    main_interpreter.allocate_tensors()

    im = Image.open(path_image)

    detection_result_image = run_odt_and_draw_results(
        path_image,
        main_interpreter,
        threshold=DETECTION_THRESHOLD
    )

    # Show the detection result
    Image.fromarray(detection_result_image)

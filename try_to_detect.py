# TODO: 使用OpenVINO对训练好的模型进行推理

from ultralytics import YOLO
import numpy as np
import cv2
from openvino.runtime import Core

# def parse_yolo_output(output, conf_threshold=0.5):
#     class_ids = []
#
#     num_predictions = output.shape[1]  # 42
#     num_values_per_prediction = output.shape[2]  # 525
#     class_id = 4
#     for i in range(num_predictions):
#         if i <= 3:
#             continue
#         prediction = output[0, i, :]
#         conf = prediction[4]
#         print(conf)
#         if conf > output[0, class_id, :][4]:
#             class_id = i
#         # class_probs = prediction[5:]
#         # print(conf)
#         # class_id = np.argmax(class_probs)
#         # confidence = class_probs[class_id] * conf
#         #
#         # if confidence > conf_threshold:
#         #     class_ids.append(class_id)
#     print(i)
#     return class_id

# -------- Step 1. Load the model --------
core = Core()
model = core.read_model(model="runs/detect/train/weights/best.xml")
original_model = YOLO('runs/detect/train/weights/best.pt')
compiled_model = core.compile_model(model=model, device_name="CPU")

# -------- Step 2. Create an inference request --------
infer_request = compiled_model.create_infer_request()

# -------- Step 3. Prepare input data --------
image = cv2.imread('test_images/test1.jpg')
input_image = cv2.resize(image, (160, 160))
input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
input_image = input_image[np.newaxis, :]  # Add batch dimension
input_tensor = infer_request.get_input_tensor()
input_tensor.data[:] = input_image

# -------- Step 4. Start inference --------
infer_request.infer()

# -------- Step 5. Get the inference result --------
original_output = original_model(image)
output_tensor = infer_request.get_output_tensor(0)
output_shape = output_tensor.shape
print(f"The shape of output tensor: {output_shape}")

# -------- Step 6. Postprocess the result --------
output_buffer = output_tensor.data[:]
output = np.array(output_buffer).reshape(output_shape)

class_ids = np.argmax(output)
print(class_ids)

# name = ['Abyssinian','American Bulldog','American Pit Bull Terrier', 'Basset Hound', 'Beagle', 'Bengal', 'Birman',
#         'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel',
#         'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond',
#         'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'RagDoll',
#         'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx',
#         'Staffordshire Bull Terrie','Wheaten Terrier','Yorkshire Terrier']
# print(f"ID: {name[class_ids]}")


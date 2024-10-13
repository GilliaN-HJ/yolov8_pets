# TODO: 将训练好的最优模型best.pt转换为.onnx格式，再调用Model Optimizer API将.onnx转换为可读的IR格式

from ultralytics import YOLO
from openvino.runtime import Core
from openvino.runtime import serialize

# load best.pt
model = YOLO('runs/detect/train/weights/best.pt')

# output best.onnx
model.export(format='onnx')
print("It's already converted to onnx.")

# load best.onnx
ie = Core()
onnx_model_path = 'runs/detect/train/weights/best.onnx'
model_onnx = ie.read_model(model=onnx_model_path)
serialize(model=model_onnx, xml_path="runs/detect/train/weights/best.xml",
          version="UNSPECIFIED")

print("It's already converted to xml.")

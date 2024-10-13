# TODO: 训练模型

from ultralytics import YOLO

# load
model = YOLO('yolov8n.pt')

# train
# 用cpu训练实在是太慢了。。。只能尽量少训练几轮。。。
# 训练好的模型将保存在/run/detect/train中
trained_model = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=160,
    batch=64,
    device='cpu'
)
print("It has already trained.")
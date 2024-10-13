# TODO: 将Oxford-IIIT Pet Dataset的.xml标注转换为YOLO可用的格式（.txt），并分割训练集和验证集

import os
import shutil
from lxml import etree
import random

images_path = "E:/Oxford-IIIT Pet Dataset/images"
annotations_path = "E:/Oxford-IIIT Pet Dataset/annotations/xmls"
trainval_file = "E:/Oxford-IIIT Pet Dataset/annotations/trainval.txt"
train_image_path = "E:/3rd_rc_homework/dataset/train/images"
train_txt_path = "E:/3rd_rc_homework/dataset/train/labels"
val_image_path = "E:/3rd_rc_homework/dataset/val/images"
val_txt_path = "E:/3rd_rc_homework/dataset/val/labels"

os.makedirs(train_image_path, exist_ok=True)
os.makedirs(train_txt_path, exist_ok=True)
os.makedirs(val_image_path, exist_ok=True)
os.makedirs(val_txt_path, exist_ok=True)


def get_classes(xml_folder):
    class_set = set()
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            tree = etree.parse(os.path.join(xml_folder, xml_file))
            for obj in tree.xpath('//object'):
                class_set.add(obj.find('name').text)
    return list(class_set)


classes = get_classes(annotations_path)
class_dict = {name: i for i, name in enumerate(classes)}


# convert XML to TXT
def convert_xml_to_txt(xml_file, txt_folder, class_dict, class_name):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    txt_file = os.path.join(txt_folder, os.path.basename(xml_file).replace('.xml', '.txt'))
    with open(txt_file, 'w') as f:
        for obj in root.findall('object'):
            # class_name = obj.find('name').text
            # class_id = class_dict[class_name]
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            f.write(f"{class_name} {x_center} {y_center} {bbox_width} {bbox_height}\n")


with open(trainval_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    image_name = parts[0] + '.jpg'
    xml_name = parts[0] + '.xml'
    # split = int(parts[2])
    split = random.uniform(0, 1)  # 用随机数划分测试集和验证集
    class_name = int(parts[1])

    image_path = os.path.join(images_path, image_name)
    xml_path = os.path.join(annotations_path, xml_name)

    # 官网直接下载的images与annotation文件有些不能一一对应，直接跳过
    if not os.path.exists(image_path) or not os.path.exists(xml_path):
        print(f"skip {image_name} or {xml_name}")
        continue

    # 按训练集和验证集分开保存 训练集:验证集==8:2
    if split >= 0.2:
        shutil.copy(image_path, os.path.join(train_image_path, image_name))
        convert_xml_to_txt(xml_path, train_txt_path, class_dict, class_name)
    else:
        shutil.copy(image_path, os.path.join(val_image_path, image_name))
        convert_xml_to_txt(xml_path, val_txt_path, class_dict, class_name)

print("It's ok!")




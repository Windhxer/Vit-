import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model

"""
main函数主要完成以下操作：

设置设备，如果可用的话使用GPU
定义数据的预处理/数据增强操作
读取图片路径并打开图片
对图片进行预处理
增加batch维度
加载标签映射文件
创建模型实例
加载模型权重
运行模型进行预测
打印预测结果，并在图像上显示预测结果
"""
def main():
    # 设置设备，如果可用的话使用GPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 定义数据的预处理/数据增强操作
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 读取图片路径
    img_path = "D:/Code/deep-learning-for-image-processing/flower_data/flower_photos/tulips/10791227_7168491604.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    # 对图片进行预处理
    img = data_transform(img)

    # 增加batch维度
    img = torch.unsqueeze(img, dim=0)

    # 加载标签映射文件
    json_path = 'D:/CODE/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型实例
    model = create_model(num_classes=5, has_logits=False).to(device)

    # 加载模型权重
    model_weight_path = "D:/CODE/weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # 前向传播计算输出
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # 打印预测结果
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))
    plt.show()

if __name__ == '__main__':
    main()

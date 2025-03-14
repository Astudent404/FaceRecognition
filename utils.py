import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

def draw_faces(image, boxes, color=(255, 0, 0), thickness=2):
    """在图像上绘制人脸边界框
    
    Args:
        image: PIL Image对象或numpy数组
        boxes: 人脸边界框坐标列表，每个框为[x1, y1, x2, y2]
        color: 边界框颜色，默认红色
        thickness: 边界框线条粗细
        
    Returns:
        带有边界框的PIL Image对象
    """
    # 如果输入是numpy数组，转换为PIL Image
    if isinstance(image, np.ndarray):
        # 如果是BGR格式（OpenCV格式），转换为RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    # 创建可绘制对象
    draw = ImageDraw.Draw(image)
    
    # 绘制每个边界框
    for box in boxes:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], 
                      outline=color, width=thickness)
    
    return image

def show_image(image, title="Face Detection Result"):
    """显示图像
    
    Args:
        image: PIL Image对象或numpy数组
        title: 图像标题
    """
    if isinstance(image, np.ndarray):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_image(image, path):
    """保存图像
    
    Args:
        image: PIL Image对象或numpy数组
        path: 保存路径
    """
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    image.save(path)
    print(f"已保存结果至: {path}")
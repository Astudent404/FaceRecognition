import torch 
import torchvision
from torch import nn
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import argparse
from utils import draw_faces, show_image, save_image

class FaceDetector:
    def __init__(self, device=None):
        """初始化人脸检测器
        
        Args:
            device: 计算设备，可以是'cuda'或'cpu'
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 初始化MTCNN模型
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],  # MTCNN的三个阶段的阈值
            factor=0.709, 
            post_process=True,
            device=self.device
        )
    
    def detect_faces(self, image):
        """检测图像中的人脸
        
        Args:
            image: PIL Image对象或numpy数组
            
        Returns:
            boxes: 人脸边界框列表，每个为[x1, y1, x2, y2]
            probs: 置信度列表
        """
        # 如果是numpy数组，转换为PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 使用MTCNN检测人脸
        boxes, probs = self.mtcnn.detect(image)
        
        # 如果没有检测到人脸
        if boxes is None:
            return [], []
        
        return boxes, probs
    
    def process_image(self, image_path, output_path=None, show=True):
        """处理单张图像
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径，若为None则不保存
            show: 是否显示结果
            
        Returns:
            带有人脸标记的图像
        """
        # 读取图像
        image = Image.open(image_path)
        
        # 检测人脸
        boxes, probs = self.detect_faces(image)
        
        if len(boxes) == 0:
            print("未检测到人脸")
            return image
        
        # 打印检测结果
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            print(f"人脸 {i+1}: 位置={box}, 置信度={prob:.4f}")
        
        # 绘制人脸边界框
        result_image = draw_faces(image, boxes)
        
        # 显示结果
        if show:
            show_image(result_image, f"检测到 {len(boxes)} 个人脸")
        
        # 保存结果
        if output_path:
            save_image(result_image, output_path)
        
        return result_image
    
    def process_video(self, video_path=0, output_path=None, show=True):
        """处理视频或摄像头流
        
        Args:
            video_path: 视频路径或摄像头索引(默认为0表示使用默认摄像头)
            output_path: 输出视频路径，若为None则不保存
            show: 是否显示结果
        """
        # 打开视频或摄像头
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("无法打开视频源")
            return
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 设置视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            while True:
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测人脸
                boxes, _ = self.detect_faces(frame)
                
                # 绘制人脸边界框
                if len(boxes) > 0:
                    for box in boxes:
                        # 将浮点数转换为整数
                        box = [int(b) for b in box]
                        # OpenCV使用BGR颜色格式，所以是蓝色、绿色、红色的顺序
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                
                # 显示结果
                if show:
                    cv2.imshow('Face Detection', frame)
                    # 按'q'退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 写入视频
                if writer:
                    writer.write(frame)
        
        finally:
            # 释放资源
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='人脸检测程序')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--video', type=str, help='输入视频路径')
    parser.add_argument('--camera', action='store_true', help='使用摄像头')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--no-show', action='store_true', help='不显示结果')
    
    args = parser.parse_args()
    
    # 初始化人脸检测器
    detector = FaceDetector()
    
    # 处理输入
    if args.image:
        detector.process_image(args.image, args.output, not args.no_show)
    elif args.video:
        detector.process_video(args.video, args.output, not args.no_show)
    elif args.camera:
        detector.process_video(0, args.output, not args.no_show)
    else:
        parser.print_help()
        print("\n请指定输入源(--image, --video, --camera)!")

if __name__ == "__main__":
    main()

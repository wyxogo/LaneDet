import os
import cv2
import paddle
import numpy as np
# from tqdm import tqdm


class UFLD:
    '''
    Ultra-Fast-Lane-Detection Inference Class

    config: The config of the model, including 'model_path', 'row_anchor', 'griding_num' and 'cls_num_per_lane'

    mean: Image preprocess mean

    std: Image preprocess std
    '''

    def __init__(self, config, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.griding_num = config['griding_num']
        self.row_anchor = config['row_anchor']
        self.cls_num_per_lane = config['cls_num_per_lane']

        self.mean = mean
        self.std = std

        col_sample = np.linspace(0, 800 - 1, config['griding_num'])
        self.col_sample_w = col_sample[1] - col_sample[0]
        self.idx = (paddle.arange(
            end=config['griding_num']) + 1).reshape((-1, 1, 1))

        self.model = paddle.jit.load(
            path=config['model_path'], model_filename='__model__', params_filename='__params__')
        self.model.eval()

    def preprocess(self, frame):
        '''
        preprocess the frame into the model input

        frame: Frame

        return: Model input
        '''
        img = cv2.resize(frame, dsize=(800, 288),
                         interpolation=cv2.INTER_LINEAR)
        img = img/255.0
        img = (img-self.mean) / self.std
        img = img.transpose((2, 0, 1))
        img = img[np.newaxis, ...]
        img = paddle.to_tensor(img, dtype='float32')
        return img

    def postprocess(self, frame, result):
        '''
        postprocess the output into the visual result

        frame: Frame

        result: Model output

        return: visual result
        '''
        result = result[0][:, ::-1, :]
        prob = paddle.nn.functional.softmax(result[:-1, :, :], axis=0)
        loc = paddle.sum(prob * self.idx, axis=0).numpy()
        result = paddle.argmax(result, axis=0).numpy()
        loc[result == self.griding_num] = 0

        for i in range(loc.shape[1]):
            if np.sum(loc[:, i] != 0) > 2:
                for k in range(loc.shape[0]):
                    if loc[k, i] > 0:
                        x = int(loc[k, i] * self.col_sample_w * self.width / 800) - 1
                        y = int(self.height * (self.row_anchor[self.cls_num_per_lane - 1 - k] / 288)) - 1
                        point = (x, y)
                        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        return frame

    def DetImg(self, img):
        # 数据预处理

        self.height = img.shape[0]
        self.width = img.shape[1]

        inputs = self.preprocess(img)

        # 模型前向计算
        result = self.model(inputs)

        # 结果可视化
        vis = self.postprocess(img, result)

        return vis

    def DetVideo(self, video, save_path="./output.mp4"):
        '''
        Lane Detection

        video_path: The path of the video

        save_path: The save path of the visual result

        '''
        # 获取视频流
        cap = cv2.VideoCapture(video)
        
        # 获取视频流参数
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_size = (self.width, self.height)
        

        # 检查路径
        # output_video_path = "./output.avi"
        path, filename = os.path.split(save_path)
        if not os.path.exists(path) and path:
            os.makedirs(path)
        
        # 获取视频流参数
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 创建输出视频
        video_writer = cv2.VideoWriter(
            save_path, int(fourcc), fps, frame_size)
        
        # 模型推理
        if cap.isOpened():
            while cap.isOpened():
                # 读取视频帧
                success, frame = cap.read()
                if not success:
                    break
                
                # 数据预处理
                inputs = self.preprocess(frame)

                # 模型前向计算
                result = self.model(inputs)

                # 结果可视化
                vis = self.postprocess(frame, result)
                
                # 写入输出视频
                video_writer.write(vis)
            
        
        # 释放视频流
        cap.release()
        video_writer.release()
        return save_path

if __name__=="__main__":
    '''
    Default configurations
    '''
    Culane = {
        'model_path': 'pretrained_models/UFLD_Culane',
        'row_anchor':  [121, 131, 141, 150, 160, 170, 180, 189, 199,
                        209, 219, 228, 238, 248, 258, 267, 277, 287],
        'griding_num': 200,
        'cls_num_per_lane': 18
    }

    Tusimple = {
        'model_path': 'pretrained_models/UFLD_Tusimple',
        'row_anchor':  [64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112, 116,
                        120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172,
                        176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228,
                        232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284],
        'griding_num': 100,
        'cls_num_per_lane': 56
    }
    ufld = UFLD(config=Culane)
    ufld.DetImg("driver_37_30frame_05181432_0203_02250.jpg")
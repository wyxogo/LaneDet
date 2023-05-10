# import os
import cv2
import paddle
import numpy as np
# from tqdm import tqdm
import argparse

import gradio as gr
from infer import UFLD

def parse_args(known=False):
    parser = argparse.ArgumentParser(description="Gradio Lane Detection")
    # parser.add_argument("--model_type", "-mt", default="online", type=str, help="model type")
    parser.add_argument("--source", "-src", default="upload", type=str, help="image input source")
    parser.add_argument("--source_video", "-src_v", default="upload", type=str, help="video input source")
    parser.add_argument("--img_tool", "-it", default="editor", type=str, help="input image tool")
    parser.add_argument("--model_name", "-mn", default="Culane", type=str, help="model name")
    parser.add_argument("--server_port", "-sp", default=7861, type=int, help="server port")
    
    parser.add_argument(
        "--device",
        "-dev",
        default="cuda:0",
        type=str,
        help="cuda or cpu",
    )
    parser.add_argument(
        "--is_share",
        "-is",
        action="store_true",
        default=True,
        help="is login",
    )
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args

def get_model(model_name):
    # assert 
    if model_name=="Culane":
        return {'model_path': 'pretrained_models/UFLD_Culane',
                'row_anchor':  [121, 131, 141, 150, 160, 170, 180, 189, 199,
                                209, 219, 228, 238, 248, 258, 267, 277, 287],
                'griding_num': 200,
                'cls_num_per_lane': 18}
    elif model_name=="Tusimple":
        return {'model_path': 'pretrained_models/UFLD_Tusimple',
                'row_anchor':  [64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112, 116,
                                120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172,
                                176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228,
                                232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284],
                'griding_num': 100,
                'cls_num_per_lane': 56}

def get_image(img, model_name):
    model = get_model(model_name="Culane")
    det = UFLD(model)
    res = det.DetImg(img)
    return res

def get_video(video, model_name):
    model = get_model(model_name="Culane")
    det = UFLD(model)
    res = det.DetVideo(video)
    return res

def main(args):
    title = "车道线检测"
    description = "<div align='center'>长江大学-电子信息学院</div>"
    
    inputs_img = gr.Image(label="原始图片")
    inputs_model = gr.Dropdown(choices=["Culane", "Tusimple"],value=args.model_name, label="模型")
    inputs_img_list=[inputs_img, inputs_model]

    outputs_img = gr.Image(label="检测图片")
    outputs_img_list = [outputs_img]


    inputs_video = gr.Video(label="原始视频")  # webcam
    inputs_model = gr.Dropdown(choices=["Culane", "Tusimple"], value=args.model_name, label="模型")
    inputs_video_list=[inputs_video, inputs_model]

    outputs_video = gr.Video(label="检测视频")
    outputs_video_list = [outputs_video]


    # 接口
    det_img = gr.Interface(
        fn=get_image,
        inputs=inputs_img_list,
        outputs=outputs_img_list,
        title=title,
        description=description,
        # article=article,
        # examples=examples_img,
        # cache_examples=False,
        # theme="seafoam",
        # live=True, # 实时变更输出
        # flagging_dir="output/image",  # 输出目录
        # allow_flagging="manual",
        # flagging_options=["good", "generally", "bad"],
    )

    # det_video = gr.Interface(fn=get_video, inputs=gr.Video(), outputs=gr.Video(),live=True)
    det_video = gr.Interface(
        fn=get_video,
        inputs=inputs_video_list,
        outputs=outputs_video_list,
        title=title,
        description=description,
        # article=article,
        # examples=examples_video,
        # cache_examples=False,
        # theme="seafoam",
        # live=True, # 实时变更输出
        flagging_dir="output/image",  # 输出目录
        # allow_flagging="manual",
        # flagging_options=["good", "generally", "bad"],
    )

    det_app = gr.TabbedInterface(interface_list=[det_img, det_video], tab_names=["图片模式", "视频模式"])


    det_app.launch(
                # inbrowser=True,  # 自动打开默认浏览器
                # show_tips=True,  # 自动显示gradio最新功能
                share=args.is_share,  # 项目共享，其他设备可以访问
                # favicon_path="./icon/logo.ico",  # 网页图标
                # show_error=True,  # 在浏览器控制台中显示错误信息
                # quiet=True,  # 禁止大多数打印语句
                server_port=args.server_port,
            )

if __name__=="__main__":
    gr.close_all()
    args = parse_args()
    main(args)
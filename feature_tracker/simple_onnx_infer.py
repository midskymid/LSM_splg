import numpy as np
import onnxruntime as ort
import os
import tqdm
import cv2
import time
import matplotlib.pyplot as plt

from utils.onnx_tools import warp_corners_and_draw_matches, resize_image
from utils.xfeat_onnx_extractor import XFEAT_ONNX_Extractor
from utils.xfeat_onnx_matcher import XFEAT_ONNX_Matcher

def onnx_infer_test(extractor, matcher, test_image_path0: str, test_image_path1: str):
    """onnx_infer_test
       Inputs: extractor: ort.InferenceSession; matcher: ort.InferenceSession; test_image_path0: str; test_image_path1: str"""
    test_img0 = cv2.imread(test_image_path0, cv2.IMREAD_COLOR)
    test_img1 = cv2.imread(test_image_path1, cv2.IMREAD_COLOR)

    input_img0, resize_img0, resize_scale0 = resize_image(test_img0, (480, 640))
    input_img1, resize_img1, resize_scale1 = resize_image(test_img1, (480, 640))

    # image infer  
    mkpts0, _, feats0 = extractor(input_img0)
    mkpts1, _, feats1 = extractor(input_img1)

    # matcher infer
    matches = matcher(feats0, feats1)

    # data process
    idx0 = matches[:,0]
    idx1 = matches[:,1]
    mkpts_0 = mkpts0.reshape(-1, 2)[idx0]
    mkpts_1 = mkpts1.reshape(-1, 2)[idx1]

    canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, resize_img0, resize_img1)
    plt.figure(figsize=(12,12))
    plt.imshow(canvas[..., ::-1])
    plt.show()

def onnx_infer_speed_test(extractor, matcher, test_image_path0: str, test_image_path1: str):
    """onnx_infer_speed_test
       Inputs: extractor: ort.InferenceSession; matcher: ort.InferenceSession; test_image_path0: str; test_image_path1: str"""
    test_img0 = cv2.imread(test_image_path0, cv2.IMREAD_COLOR)
    test_img1 = cv2.imread(test_image_path1, cv2.IMREAD_COLOR)
    input_img0, resize_img0, resize_scale0 = resize_image(test_img0, (480, 640))
    input_img1, resize_img1, resize_scale1 = resize_image(test_img1, (480, 640))
    consume_times = []

    for i in tqdm.tqdm(range(1000)):
        start_time = time.time()
        
        # image infer
        
        mkpts0, sc0, feats0 = extractor(input_img0)
        mkpts1, sc1, feats1 = extractor(input_img1)
            
       # matcher infer

        matches = matcher(feats0, feats1)
        
        consume_times.append(time.time() - start_time)
    
    print(f"Average FPS per image: {1/np.mean(consume_times):.4f}")
    print("finished speed test...")



if __name__ == "__main__":
    extractor_onnx_path = "weights/480_640_1024_extractor_sim.onnx"
    matcher_onnx_path = "weights/480_640_1024_matcher_sim.onnx"
    postprocess_onnx_path = "weights/480_640_1024_postprocess_sim.onnx"
    extractor_trt_cache_path = "./trt_engine_cache/extractor"
    matcher_trt_cache_path = "./trt_engine_cache/matcher"
    postprocess_trt_cache_path = "./trt_engine_cache/postprocess"
    top_k = 1024

    test_image_path0 = "assets/aqualoc_pic1.jpg"
    test_image_path1 = "assets/aqualoc_pic2.jpg"

    # Load a model for extracting feature points from images
    extractor = XFEAT_ONNX_Extractor(extractor_onnx_path, postprocess_onnx_path, extractor_trt_cache_path, postprocess_trt_cache_path, top_k)  
    
    # Load a model for feature point matching
    matcher = XFEAT_ONNX_Matcher(matcher_onnx_path, matcher_trt_cache_path)

    onnx_infer_test(extractor, matcher, test_image_path0, test_image_path1)
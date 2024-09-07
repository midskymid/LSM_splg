import numpy as np
import onnxruntime as ort
import cv2
import torch
import os
from typing import List, Optional, Union

def print_gpu_usage():
    """To display GPU usage when the program runs to a specified position"""
    gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3  # transfer to GB
    gpu_memory_cached = torch.cuda.memory_reserved() / 1024 ** 3
    
    print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
    print(f"GPU Memory Cached: {gpu_memory_cached:.2f} GB")

def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: Optional[str] = "max",
    interp: Optional[str] = "area",
) :
    """Resize an image to a fixed size, or according to max or min edge."""
    """
    Inputs:
    size: (desired_width, desired_height): if you give an int value, it will resize image depends on image's max or min edge scale; If your parameter type is a list or tuple, it will try to resize the image to the specified size.
    fn: resize method(it can be used when size's type is int).
    interp: resize function.
    Outputs:
    input_image: np.ndarray: size -> (1, C, H, W).
    resize_image: np.ndarray: size -> (H, W, C).
    scale: tuple: scale = (w_new / w_src, h_new / h_src).
    """
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    
    resize_image = cv2.resize(image, (w_new, h_new), interpolation=mode)
    input_image = np.expand_dims(resize_image.transpose(2, 0, 1).astype(np.float32), axis=0)
    return input_image, resize_image, scale

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches

def print_onnx_model_io(onnx_model_path: str):
    # Parse model file
    if not os.path.exists(onnx_model_path):
        print("ONNX file {} not found.".format(onnx_model_path))
        return None

    tmp_ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    # print the input,output names and shapes
    for i in range(len(tmp_ort_session.get_inputs())):
        print(f"Input name: {tmp_ort_session.get_inputs()[i].name}, shape: {tmp_ort_session.get_inputs()[i].shape}, type: {type(tmp_ort_session.get_inputs()[i])}")
    for i in range(len(tmp_ort_session.get_outputs())):
        print(f"Output name: {tmp_ort_session.get_outputs()[i].name}, shape: {tmp_ort_session.get_outputs()[i].shape}, type: {type(tmp_ort_session.get_outputs()[i])}")

def create_ort_session(onnx_model_path: str, trt_engine_cache_path: str ='trt_engine_cache', try_tensorrt: bool =True):
    # Parse model file
    if not os.path.exists(onnx_model_path):
        print("ONNX file {} not found.".format(onnx_model_path))
        return None
    
    # only provide TensorRT inference method
    providers = [
    ('TensorrtExecutionProvider', { 
        'device_id': 0,
        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': trt_engine_cache_path,
        'trt_dump_subgraphs': False,
        
    }),
    ('CUDAExecutionProvider', { 
        'device_id': 0,
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
    }),
    ('CPUExecutionProvider',{ 
    })
]
    if not try_tensorrt:
        providers = providers[1:]
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers)

    return ort_session




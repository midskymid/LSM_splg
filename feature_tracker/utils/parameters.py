import cv2
import os


def readParameters(path: str) -> dict:
    """read parameters from config file and return dictionary of parameters
       Input: type(str) : your config file which format is OpenCV YAML.
       Output: type(dict) : Visual-Inertial Odometry related parameters, excluding parameters related to nonlinear optimization.
    """
    params = {}
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found at {path}, please check the config file path...")
    if not path.lower().endswith('.yaml'):
        raise ValueError(f"{path} is not YAML format...")
    try:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    
        params['image_topic'] = fs.getNode('image_topic').string()
        params['imu_topic'] = fs.getNode('imu_topic').string()
        params['sp_uw_model_path'] = fs.getNode('sp_uw_model_path').string()
        params['sp_lg_model_path'] = fs.getNode('sp_lg_model_path').string()
        params['extractor_onnx_path'] = fs.getNode('extractor_onnx_path').string()
        params['matcher_onnx_path'] = fs.getNode('matcher_onnx_path').string()
        params['postprocess_onnx_path'] = fs.getNode('postprocess_onnx_path').string()
        params['extractor_engine_path'] = fs.getNode('extractor_engine_path').string()
        params['matcher_engine_path'] = fs.getNode('matcher_engine_path').string()
        params['postprocess_engine_path'] = fs.getNode('postprocess_engine_path').string()

        params['top_k'] = int(fs.getNode('top_k').real())
        params['max_cnt'] = int(fs.getNode('max_cnt').real())
        params['min_dist'] = int(fs.getNode('min_dist').real())
        params['image_height'] = int(fs.getNode('image_height').real())
        params['image_width'] = int(fs.getNode('image_width').real())
        params['freq'] = int(fs.getNode('freq').real())
        params['F_threshold'] = fs.getNode('F_threshold').real()
        params['show_track'] = int(fs.getNode('show_track').real())
        params['equalize'] = fs.getNode('equalize').real()
        
        params['extrinsicRotation'] = fs.getNode('extrinsicRotation').mat()
        params['extrinsicTranslation'] = fs.getNode('extrinsicTranslation').mat()
          
        params['distortion_parameters'] = [fs.getNode('distortion_parameters').getNode(key).real() for key in fs.getNode('distortion_parameters').keys()]
        params['projection_parameters'] = [fs.getNode('projection_parameters').getNode(key).real() for key in fs.getNode('projection_parameters').keys()]
        
        return params
          
    except Exception as e:
        print("Error loading YAML: ", e)
        
        
if __name__ == "__main__":
    # change your config file path to check
    path = "your_config_yaml_file_path.yaml"
    params = readParameters(path)
    print("The paramters are as follows: \n", params)
    
        

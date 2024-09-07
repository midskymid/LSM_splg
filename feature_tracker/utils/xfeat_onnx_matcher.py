from .onnx_tools import *

class XFEAT_ONNX_Matcher:

    def __init__(self, ModelPath: str, trt_engine_cache_path: str) -> None:
        if not (os.path.exists(ModelPath) and ModelPath.lower().endswith('.onnx')):
            raise FileNotFoundError(f"onnx file not found at {ModelPath}, please check the path...")
        
        # if not (os.path.exists(trt_engine_cache_path)):
        #     raise FileNotFoundError(f"TensorRT engine cache path does not exist at {trt_engine_cache_path}, please check the path...")
        
        self.matcher_ort_session = create_ort_session(ModelPath, trt_engine_cache_path)

    def inference_preprocessed(self, feats0, feats1):
        # organize inputs
        inputs_dict = {
            self.matcher_ort_session.get_inputs()[0].name: feats0, 
            self.matcher_ort_session.get_inputs()[1].name: feats1
        }

        # do inference
        onnx_outputs = self.matcher_ort_session.run(None, inputs_dict)

        # organize outputs
        matches = onnx_outputs[0].astype(np.int32)
        
        return matches

        
    def __call__(self, feats0: np.ndarray, feats1: np.ndarray):
        return self.inference_preprocessed(feats0, feats1)
from .onnx_tools import *

class XFEAT_ONNX_Extractor:

    def __init__(self, DetectModelPath: str, PostModelPath : str, detect_trt_engine_cache_path: str, post_trt_engine_cache_path : str, top_k: int = 1024) -> None:
        if not (os.path.exists(DetectModelPath) and os.path.exists(PostModelPath) and DetectModelPath.lower().endswith('.onnx') and PostModelPath.lower().endswith('.onnx')):
            raise FileNotFoundError(f"onnx file not found at {DetectModelPath} or {PostModelPath}, please check the path...")
        
        # if not (os.path.exists(detect_trt_engine_cache_path) and post_trt_engine_cache_path):
        #     raise FileNotFoundError(f"TensorRT engine cache path does not exist at {detect_trt_engine_cache_path} or {post_trt_engine_cache_path}, please check the path...")
        
        self.top_k = np.array(top_k, dtype=np.int32)
        # 预先分配top_k的ortvalue，位于cuda:0
        self.top_k_ort_value = ort.OrtValue.ortvalue_from_numpy(self.top_k, 'cuda', 0)

        self.xfeat_ort_session = create_ort_session(DetectModelPath, detect_trt_engine_cache_path, try_tensorrt=True)
        self.postprocess_ort_session = create_ort_session(
            PostModelPath, 
            post_trt_engine_cache_path,
            try_tensorrt=True
        )

        # 创建主模型的io_binding
        self.xfeat_ort_session_io_binding = self.xfeat_ort_session.io_binding()
        # 预先分配 xfeat 的 M1 输出，位于 cuda:0
        self.xfeat_output_2_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type(self.xfeat_ort_session.get_outputs()[2].shape, np.float32, 'cuda', 0)
        # 指定 xfeat 的 mkpts, scores 输出，位于 cuda:0
        self.xfeat_ort_session_io_binding.bind_output(self.xfeat_ort_session.get_outputs()[0].name, 'cuda', 0)
        self.xfeat_ort_session_io_binding.bind_output(self.xfeat_ort_session.get_outputs()[1].name, 'cuda', 0)
        # 指定 xfeat 的 M1 输出，使用预分配的 ortvalue
        self.xfeat_ort_session_io_binding.bind_ortvalue_output(self.xfeat_ort_session.get_outputs()[2].name, self.xfeat_output_2_ortvalue)

        # 创建后处理模型的io_binding
        self.postprocess_ort_session_io_binding = self.postprocess_ort_session.io_binding()
        # 指定后处理模型的 M1 输入，使用 xfeat 的输出
        self.postprocess_ort_session_io_binding.bind_ortvalue_input(self.postprocess_ort_session.get_inputs()[2].name, self.xfeat_output_2_ortvalue)
        # 指定后处理模型的 top_k 输入，使用预分配的 ortvalue
        self.postprocess_ort_session_io_binding.bind_ortvalue_input(self.postprocess_ort_session.get_inputs()[7].name, self.top_k_ort_value)
        # 预先分配后处理模型的全部输出，位于 cuda:0
        self.postprocess_output_0_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type((1, top_k, 2), np.int32, 'cuda', 0)
        self.postprocess_output_1_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type((1, top_k), np.float32, 'cuda', 0)
        self.postprocess_output_2_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type((1, top_k, 64), np.float32, 'cuda', 0)
        # 指定后处理模型的全部输出，使用预分配的 ortvalue
        self.postprocess_ort_session_io_binding.bind_ortvalue_output(self.postprocess_ort_session.get_outputs()[0].name, self.postprocess_output_0_ortvalue)
        self.postprocess_ort_session_io_binding.bind_ortvalue_output(self.postprocess_ort_session.get_outputs()[1].name, self.postprocess_output_1_ortvalue)
        self.postprocess_ort_session_io_binding.bind_ortvalue_output(self.postprocess_ort_session.get_outputs()[2].name, self.postprocess_output_2_ortvalue)

        # 从 numpy 数组创建后处理模型的输入 ortvalue，位于 cuda:0。主模型为固定分辨率，这些值是固定的，可以放到初始化中，不需要每次都创建。
        H, W = self.xfeat_ort_session.get_inputs()[0].shape[-2:]
        _H, _W = (H//32) * 32, (W//32) * 32
        rh, rw = H/_H, W/_W
        rh_ort_value = ort.OrtValue.ortvalue_from_numpy(np.array(rh, dtype=np.float32), 'cuda', 0)
        rw_ort_value = ort.OrtValue.ortvalue_from_numpy(np.array(rw, dtype=np.float32), 'cuda', 0)
        _H_ort_value = ort.OrtValue.ortvalue_from_numpy(np.array(_H, dtype=np.int32), 'cuda', 0)
        _W_ort_value = ort.OrtValue.ortvalue_from_numpy(np.array(_W, dtype=np.int32), 'cuda', 0)
        # 指定后处理模型的 rh, rw, _H, _W 输入
        self.postprocess_ort_session_io_binding.bind_ortvalue_input(self.postprocess_ort_session.get_inputs()[3].name, rh_ort_value)
        self.postprocess_ort_session_io_binding.bind_ortvalue_input(self.postprocess_ort_session.get_inputs()[4].name, rw_ort_value)
        self.postprocess_ort_session_io_binding.bind_ortvalue_input(self.postprocess_ort_session.get_inputs()[5].name, _H_ort_value)
        self.postprocess_ort_session_io_binding.bind_ortvalue_input(self.postprocess_ort_session.get_inputs()[6].name, _W_ort_value)
        

        
    def __call__(self, input_array: np.ndarray):
        # input_array = input_array.transpose(2, 0, 1).astype(np.float32)  # BGR to RGB
        # input_array = np.expand_dims(input_array, axis=0)  # add batch_size dim

        # 从 numpy 数组创建 xfeat 的输入 ortvalue，位于 cuda:0
        xfeat_input_ortvalue = ort.OrtValue.ortvalue_from_numpy(input_array, 'cuda', 0)
        # 指定 xfeat 的输入 ortvalue
        self.xfeat_ort_session_io_binding.bind_ortvalue_input(self.xfeat_ort_session.get_inputs()[0].name, xfeat_input_ortvalue)
        # 运行 xfeat
        self.xfeat_ort_session.run_with_iobinding(self.xfeat_ort_session_io_binding)

        # 获取 xfeat 的 ortvalue 输出，位于 cuda:0，无需拷贝到cpu
        xfeat_ort_session_outputs = self.xfeat_ort_session_io_binding.get_outputs()
        # 指定后处理模型的 mkpts, scores_in 输入，使用 xfeat 的输出
        self.postprocess_ort_session_io_binding.bind_ortvalue_input(self.postprocess_ort_session.get_inputs()[0].name, xfeat_ort_session_outputs[0])
        self.postprocess_ort_session_io_binding.bind_ortvalue_input(self.postprocess_ort_session.get_inputs()[1].name, xfeat_ort_session_outputs[1])

        # 运行后处理模型
        self.postprocess_ort_session.run_with_iobinding(self.postprocess_ort_session_io_binding)

        # 获取后处理模型的全部输出，拷贝到cpu
        outputs = self.postprocess_ort_session_io_binding.copy_outputs_to_cpu()

        keypoints = outputs[0]
        scores = outputs[1]
        descriptors = outputs[2]      

        return keypoints, scores, descriptors

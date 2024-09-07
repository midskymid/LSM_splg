import types
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxsim

from modules.xfeat import XFeat
from modules.interpolator import InterpolateSparse2d

class CustomInstanceNorm(torch.nn.Module):
    def __init__(self, epsilon=1e-5):
        super(CustomInstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), unbiased=False, keepdim=True)
        return (x - mean) / (std + self.epsilon)
    
class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """ 
    def __init__(self, mode = 'bicubic', align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        if torch.onnx.is_in_onnx_export():
            x1 = x[...,0] / (W-1)
            x2 = x[...,1] / (H-1)
            return 2. * torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1)], dim = -1) - 1.
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)

        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)


class PostProcess(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.interpolator = InterpolateSparse2d('bicubic')

    def forward(self, mkpts, scores, M1, rw1, rh1, _H1, _W1, top_k):

        idxs = torch.topk(scores, top_k, dim=-1)[1]
        mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)
        mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)
        mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)

        feats = self.interpolator(M1, mkpts, H = _H1, W = _W1)

        #L2-Normalize
        feats = F.normalize(feats, dim=-1)

        mkpts[..., 0] = mkpts[..., 0] * rw1
        mkpts[..., 1] = mkpts[..., 1] * rh1

        return [{'keypoints': mkpts, 'scores': scores, 'descriptors': feats}]
    
@torch.inference_mode()
def match(feats1, feats2):
    feats1 = feats1.reshape(-1, 64)
    feats2 = feats2.reshape(-1, 64)
    cossim = feats1 @ feats2.t()
    cossim_t = feats2 @ feats1.t()
    
    _, match12 = cossim.max(dim=1)
    _, match21 = cossim_t.max(dim=1)

    idx0 = torch.arange(match12.shape[0], device=match12.device)
    mutual = match21[match12] == idx0

    idx0 = idx0[mutual]
    idx1 = match12[mutual]

    return idx0, idx1

@torch.inference_mode()
def match_xfeat(self, feats0, feats1):
    #Match batches of pairs
    idxs0, idxs1 = match(feats0,feats1)
    matches = torch.zeros((idxs0.shape[0], 2))
    matches[:,0] = idxs0
    matches[:,1] = idxs1

    return matches

def parse_args():
    parser = argparse.ArgumentParser(description="Export XFeat/Matching model to ONNX.")
    
    parser.add_argument(
        "--split_instance_norm",
        action="store_true",
        help="Whether to split InstanceNorm2d into '(x - mean) / (std + epsilon)', due to some inference libraries not supporting InstanceNorm, such as OpenVINO.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Input image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Input image width.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1024,
        help="Keep best k features.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic axes.",
    )
    parser.add_argument(
        "--dir_export_path",
        type=str,
        default="weights/",
        help="Directory of path to export ONNX model.",
    )
    parser.add_argument(
        "--extractor_export_path",
        type=str,
        default="extractor.onnx",
        help="Path to export extractor ONNX model.",
    )
    parser.add_argument(
        "--postprocess_export_path",
        type=str,
        default="postprocess.onnx",
        help="Path to export postprocess ONNX model.",
    )
    parser.add_argument(
        "--matcher_export_path",
        type=str,
        default="matcher.onnx",
        help="Path to export matcher ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version.",
    )

    return parser.parse_args()

def export_onnx(
    split_instance_norm = False,
    height = 480,
    width = 640,
    top_k = 1024,
    dynamic = False,
    dir_export_path = "weights/",
    extractor_export_path = None,
    postprocess_export_path = None,
    matcher_export_path = None,
    opset = None
):
    if extractor_export_path is None or matcher_export_path is None or postprocess_export_path is None:
        raise ValueError(f"{extractor_export_path} or {matcher_export_path} or {postprocess_export_path} is None, please give it a correct path.")
    else:
        extractor_export_path = dir_export_path + str(height) + "_" + str(width) + "_" + str(top_k) + "_" + extractor_export_path
        postprocess_export_path = dir_export_path + str(height) + "_" + str(width) + "_" + str(top_k) + "_" + postprocess_export_path
        matcher_export_path = dir_export_path + str(height) + "_" + str(width) + "_" + str(top_k) +  "_" + matcher_export_path
    
    xfeat = XFeat()
    xfeat = xfeat.cpu().eval()
    xfeat.dev = "cpu"
    xfeat.top_k = top_k

    if split_instance_norm:
        xfeat.net.norm = CustomInstanceNorm()

    batch_size = 1
    x1 = torch.randn(batch_size, 3, height, width, dtype=torch.float32, device='cpu')

    # export extractor
    xfeat.forward = xfeat.detectAndCompute_backbone
    dynamic_axes = {"images"}
    torch.onnx.export(
        xfeat,
        (x1),
        extractor_export_path,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["mkpts", "scores", "M1"],
        dynamic_axes=None,
    )

    # export matcher
    xfeat.forward = types.MethodType(match_xfeat, xfeat)

    feats0 = torch.randn(batch_size, top_k, 64, dtype=torch.float32, device='cpu')
    feats1 = torch.randn(batch_size, top_k, 64, dtype=torch.float32, device='cpu')

    dynamic_axes = {
        "feats0": {1: "num_keypoints_0"},
        "feats1": {1: "num_keypoints_1"}
    }

    torch.onnx.export(
        xfeat,
        (feats0, feats1),
        matcher_export_path,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["feats0", "feats1"],
        output_names=["matches"],
        dynamic_axes=dynamic_axes if dynamic else None,
    )

    # export postprocess
    xfeat_postprocess = PostProcess()
    top_k = torch.tensor(top_k, dtype=torch.int32)
    mkpts = torch.randn(batch_size, 14409, 2, dtype=torch.float32, device='cpu').to(torch.int32)
    scores = torch.randn(batch_size, 14409, dtype=torch.float32, device='cpu')
    M1 = torch.randn(batch_size, 64, 60, 80, dtype=torch.float32, device='cpu')
    rw1 = torch.tensor(1.0, dtype=torch.float32)
    rh1 = torch.tensor(1.0, dtype=torch.float32)
    _H1 = torch.tensor(480, dtype=torch.int32)
    _W1 = torch.tensor(640, dtype=torch.int32)

    dynamic_axes = {
        "mkpts": {1: "num_keypoints_0"},
        "scores_in": {1: "num_keypoints_1"},
        "M1": {2: "feature_height", 3: "feature_width"},
    }

    torch.onnx.export(
        xfeat_postprocess,
        (mkpts, scores, M1, rh1, rw1, _H1, _W1, top_k),
        postprocess_export_path,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["mkpts", "scores_in", "M1", "rh1", "rw1", "_H1", "_W1", "top_k"],
        output_names=["keypoints", "scores", "descriptors"],
        dynamic_axes=dynamic_axes if dynamic else None,
    )

    # simplify extractor onnx model
    extractor_onnx_model = onnx.load(extractor_export_path)  # load onnx model
    onnx.checker.check_model(extractor_onnx_model)  # check onnx model
    sim_extractor_onnx_model, check = onnxsim.simplify(extractor_onnx_model)
    assert check, "simplify extractor onnx model failed."
    src_extractor_export_path_name, ext = os.path.splitext(extractor_export_path)
    extractor_export_path = src_extractor_export_path_name + "_sim" + ext
    onnx.save(sim_extractor_onnx_model, extractor_export_path)
    print(f"Extractor Onnx Model exported to {extractor_export_path}.")

    # simplify matcher onnx model
    matcher_onnx_model = onnx.load(matcher_export_path)  # load onnx model
    onnx.checker.check_model(matcher_onnx_model)  # check onnx model
    sim_matcher_onnx_model, check = onnxsim.simplify(matcher_onnx_model)
    assert check, "simplify matcher onnx model failed."
    src_matcher_export_path_name, ext = os.path.splitext(matcher_export_path)
    matcher_export_path = src_matcher_export_path_name + "_sim" + ext
    onnx.save(sim_matcher_onnx_model, matcher_export_path)
    print(f"Matcher Onnx Model exported to {matcher_export_path}.")

    # simplify postprocess onnx model
    postprocess_onnx_model = onnx.load(postprocess_export_path)
    onnx.checker.check_model(postprocess_onnx_model)
    sim_postprocess_onnx_model, check = onnxsim.simplify(postprocess_onnx_model)
    assert check, "simplify postprocess onnx model failed."
    src_postprocess_export_path_name, ext = os.path.splitext(postprocess_export_path)
    postprocess_export_path = src_postprocess_export_path_name + "_sim" + ext
    onnx.save(sim_postprocess_onnx_model, postprocess_export_path)
    print(f"postprocess Onnx Model exported to {postprocess_export_path}.")


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))



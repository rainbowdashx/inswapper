import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swapper import *
from restoration import *

model = "./checkpoints/inswapper_128.onnx"
providers = onnxruntime.get_available_providers()
face_analyser = getFaceAnalyser(model, providers)


model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
face_swapper = getFaceSwapModel(model_path)
check_ckpts()
ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
checkpoint = torch.load(ckpt_path)["params_ema"]


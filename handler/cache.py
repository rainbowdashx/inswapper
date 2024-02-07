import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('./CodeFormer/CodeFormer')

from swapper import *
from restoration import *

model = "./checkpoints/inswapper_128.onnx"
providers = onnxruntime.get_available_providers()
face_analyser = getFaceAnalyser(model, providers)


model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
face_swapper = getFaceSwapModel(model_path)
check_ckpts()
upsampler = set_realesrgan()
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                    codebook_size=1024,
                                                    n_head=8,
                                                    n_layers=9,
                                                    connect_list=["32", "64", "128", "256"],
                                                ).to(device)
ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
checkpoint = torch.load(ckpt_path)["params_ema"]
codeformer_net.load_state_dict(checkpoint)
codeformer_net.eval()


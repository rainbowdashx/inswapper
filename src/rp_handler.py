
import sys
import os

import time

import runpod
import requests
from requests.adapters import HTTPAdapter, Retry

import base64
from PIL import Image
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swapper import *
from restoration import *



automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))


def base64_to_pil_image(base64_str):
    binary_data = base64.b64decode(base64_str)
    binary_file = BytesIO(binary_data)
    img = Image.open(binary_file)
    return img

def pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')

def run_restoration(result_image):

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

    result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
    result_image = face_restoration(result_image, 
                                    args.background_enhance, 
                                    args.face_upsample, 
                                    args.upscale, 
                                    args.codeformer_fidelity,
                                    upsampler,
                                    codeformer_net,
                                    device)
    result_image = Image.fromarray(result_image)
    return result_image
    


def run_inference(inference_request):
    

    source_img = [base64_to_pil_image(inference_request["source_img"])]
    target_img = base64_to_pil_image(inference_request["target_img"])

    model = "./checkpoints/inswapper_128.onnx"

    result_image = process(source_img, target_img, -1, -1, model)

    restored_image = run_restoration(result_image)    

    restored_image_base64 = pil_image_to_base64(restored_image)

    result = {
        "image": restored_image_base64
    }

    return result

def handler(event):
    json = run_inference(event["input"])
    return json


if __name__ == "__main__":
    print("Service is ready. Starting RunPod...")
    runpod.serverless.start({"handler": handler})

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swapper import *

model = "./checkpoints/inswapper_128.onnx"
providers = onnxruntime.get_available_providers()
face_analyser = getFaceAnalyser(model, providers)
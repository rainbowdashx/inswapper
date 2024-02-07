sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swapper import *

source_img = [Image.open("./data/man1.jpeg"),Image.open("./data/man2.jpeg")]
target_img = Image.open("./data/mans1.jpeg")

model = "./checkpoints/inswapper_128.onnx"
result_image = process(source_img, target_img, -1, -1, model)
result_image.save("result.png")
# !git clone https://github.com/ultralytics/yolov5
# !pip install -r yolov5/requirements.txt

import torch
import sys
sys.path.append('../../../yolov5')

from models.common import DetectMultiBackend

model = DetectMultiBackend('yolov5s.pt', device='cuda' if torch.cuda.is_available() else 'cpu')

# freeze params for Transfer Learning
for param in model.model.parameters():
    param.requires_grad = False

# define layers for fine-tunning
for module in model.model.modules():
    if hasattr(module, 'cv2'):
        for param in module.parameters():
            param.requires_grad = True

trainable_params = filter(lambda p: p.requires_grad, model.model.parameters())
trainable_params = list(trainable_params)

if len(trainable_params) == 0:
    raise ValueError("No prarameters defined for the fine-tunning verify the layers")

optimizer = torch.optim.Adam(trainable_params, lr=0.001)

dummy_input = torch.randn(1, 3, 640, 640)
output = model(dummy_input)

print("Output Shapes:")
for i, out in enumerate(output):
    if isinstance(out, list):
        print(f"Output {i} is a list with {len(out)} elements.")
        for j, sub_out in enumerate(out):
            if hasattr(sub_out, 'shape'):
                print(f"    Element {j} Shape:", sub_out.shape)
            else:
                print(f"    Element {j} is not a tensor.")
    elif hasattr(out, 'shape'):
        print(f"Output {i} Shape:", out.shape)
    else:
        print(f"Output {i} is not a tensor.")

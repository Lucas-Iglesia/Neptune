import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# freeze parameters for transfer learning
for param in model.parameters():
    param.requires_grad = False

# unfreeze layers for fine-tuning
for module in model.modules():
    if hasattr(module, 'cv2'):
        for param in module.parameters():
            param.requires_grad = True

# verify trainable parameters
trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
if not trainable_params:
    raise ValueError("No parameters defined for fine-tuning; verify the layers configuration.")

optimizer = torch.optim.Adam(trainable_params, lr=0.001)

# Test model output with dummy input
dummy_input = torch.randn(1, 3, 640, 640).to(device)
output = model(dummy_input)

# Display output shapes
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

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import onnx

class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)[0]  # Return only the first dictionary of predictions

# 1. Load the PyTorch model
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=4)
model.load_state_dict(torch.load("fasterrcnn_mask_detector.pth", map_location=torch.device('cpu')))
model.eval()

# Wrap the model
wrapped_model = WrapperModel(model)

# 2. Create a dummy input (single tensor, not a list)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export to ONNX
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "fasterrcnn_mask_detector.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["boxes", "labels", "scores"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "boxes": {0: "batch_size"},
        "labels": {0: "batch_size"},
        "scores": {0: "batch_size"}
    }
)

# 4. Verify the ONNX model
onnx_model = onnx.load("fasterrcnn_mask_detector.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model exported and verified successfully!")

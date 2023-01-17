from torchvision.models import resnet152
from torchvision.io import read_image
import torch
from PIL import Image

model = resnet152()

scripted_module = torch.jit.script(model)
img = Image.open("my_cat.jpg").convert('RGB')

from torchvision import transforms
#
# Create a preprocessing pipeline
#
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
#
# Pass the image for preprocessing and the image preprocessed
#
img_cat_preprocessed = preprocess(img)
#
# Reshape, crop, and normalize the input tensor for feeding into network for evaluation
#
batch_img_cat_tensor = torch.unsqueeze(img_cat_preprocessed, 0)


model.eval()
scripted_module.eval()
unscript_prediction = model(batch_img_cat_tensor).squeeze(0).softmax(0)
script_prediction = scripted_module(batch_img_cat_tensor).squeeze(0).softmax(0)

unscripted_top5_indices = torch.topk(unscript_prediction, 5)
scripted_top5_indices = torch.topk(script_prediction, 5)


assert torch.allclose(unscripted_top5_indices.indices, scripted_top5_indices.indices)
from PIL import Image
from typing import List
import torch
from torchvision import transforms


def detection(input_image: Image, model):
	results = model(input_image)
	crops = results.crop(save=False)
	#return [crop.cpu().item() for crop in crops[0]['box']]
	#print(crops)
	return [i['im'] for i in crops]


def embedding_creation(crops: List, model):
	image = crops[0]['im']
	preprocess = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	input_tensor = preprocess(image)
	input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

	# move the input and model to GPU for speed if available
	if torch.cuda.is_available():
		input_batch = input_batch.to('cuda')
		model.to('cuda')

	with torch.no_grad():
		output = model.features(input_batch)
		x = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
		x = torch.flatten(x, 1)
	return x.cpu().tolist()
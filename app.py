import io
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

import torch
import torch.nn as nn
from matplotlib import cm

DEFAULT_WEIGHTS = "runs/classify/brain_tumor/weights/best.pt"
FIXED_IMGSZ = 224
DISPLAY_WIDTH = 320


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str) -> YOLO:
	if not os.path.exists(weights_path):
		raise FileNotFoundError(
			f"Weights not found at '{weights_path}'. Train the model first to generate best.pt."
		)
	model = YOLO(weights_path)
	return model


def predict_image(model: YOLO, image: Image.Image, imgsz: int = FIXED_IMGSZ):
	array = np.array(image.convert("RGB"))
	results = model.predict(source=array, imgsz=imgsz, save=False, verbose=False)
	result = results[0]
	probs = result.probs
	if probs is None:
		raise RuntimeError("Model did not return probabilities. Ensure it's a classification checkpoint.")
	scores = probs.data.tolist()
	class_names = result.names
	pred_index = int(np.argmax(scores))
	pred_label = class_names[pred_index]
	return pred_label, scores, class_names


def _find_last_conv2d(module: nn.Module) -> nn.Module:
	last_conv = None
	for m in module.modules():
		if isinstance(m, nn.Conv2d):
			last_conv = m
	return last_conv


def _toggle_silu_inplace(module: nn.Module, inplace: bool):
	"""Set all nn.SiLU(inplace=...) flags to desired value. Returns list of (module, old_value) to restore later."""
	changes = []
	for m in module.modules():
		if isinstance(m, nn.SiLU):
			changes.append((m, getattr(m, "inplace", False)))
			m.inplace = inplace
	return changes


def _restore_silu_inplace(changes):
	for m, old in changes:
		m.inplace = old


def generate_gradcam(model: YOLO, image: Image.Image, target_class_index: int, imgsz: int = FIXED_IMGSZ) -> Image.Image:
	# Resize with PIL (avoid OpenCV)
	img_rgb = np.array(image.convert("RGB"))
	resized_pil = image.convert("RGB").resize((imgsz, imgsz), resample=Image.BILINEAR)
	tensor = torch.from_numpy(np.array(resized_pil)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
	tensor.requires_grad_(True)

	model_torch = model.model
	model_torch.eval()

	target_layer = _find_last_conv2d(model_torch)
	if target_layer is None:
		return image

	activations = None
	gradients = None

	def fwd_hook(_, __, output):
		nonlocal activations
		activations = output

	def bwd_hook(_, grad_input, grad_output):
		nonlocal gradients
		gradients = grad_output[0].clone()

	fwd_handle = target_layer.register_forward_hook(fwd_hook)
	bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

	changes = _toggle_silu_inplace(model_torch, inplace=False)
	try:
		with torch.enable_grad():
			logits = model_torch(tensor)
			if isinstance(logits, (list, tuple)):
				logits = logits[0]
			target_logit = logits[:, target_class_index].squeeze()
			model_torch.zero_grad(set_to_none=True)
			target_logit.backward(retain_graph=True)
	finally:
		fwd_handle.remove()
		bwd_handle.remove()
		_restore_silu_inplace(changes)

	if activations is None or gradients is None:
		return image

	weights = gradients.mean(dim=(2, 3), keepdim=True)
	saliency = (weights * activations).sum(dim=1, keepdim=True)
	saliency = torch.relu(saliency)
	saliency = saliency.squeeze().detach().cpu().numpy()
	if saliency.max() > 0:
		saliency = saliency / saliency.max()
	else:
		saliency = np.zeros_like(saliency)

	# Create heatmap with matplotlib colormap (avoid OpenCV)
	cmap = cm.get_cmap('jet')
	heatmap = (cmap(saliency)[:, :, :3] * 255).astype(np.uint8)  # HxWx3 RGB
	# Resize heatmap to original image size
	heatmap_img = Image.fromarray(heatmap).resize((img_rgb.shape[1], img_rgb.shape[0]), resample=Image.BILINEAR)
	# Overlay
	overlay = Image.blend(Image.fromarray(img_rgb), heatmap_img, alpha=0.4)
	return overlay


st.set_page_config(page_title="Brain Tumor Detection (YOLOv8)", layout="wide")

st.title("Brain Tumor Detection with YOLOv8 (Classification)")

model = None
model_error = None
try:
	model = load_model(DEFAULT_WEIGHTS)
except Exception as e:
	model_error = str(e)

left, right = st.columns([1, 1])

with left:
	uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]) 
	if uploaded is not None:
		image = Image.open(io.BytesIO(uploaded.read()))
		st.image(image, caption="Uploaded Image", width=DISPLAY_WIDTH)
	else:
		st.info("Upload an image to get a prediction.")

with right:
	if model_error:
		st.warning(model_error)
	elif uploaded is not None and model is not None:
		pred_label, scores, class_names = predict_image(model, image, imgsz=FIXED_IMGSZ)
		label_to_index = {name.lower(): idx for idx, name in class_names.items()}
		yes_idx = label_to_index.get("yes")
		no_idx = label_to_index.get("no")

		yes_prob = scores[yes_idx] if yes_idx is not None else None
		no_prob = scores[no_idx] if no_idx is not None else (1.0 - yes_prob if yes_prob is not None else None)

		st.subheader("Result")
		show_cam = False
		cam_img = None
		if yes_prob is not None and (no_prob is None or yes_prob >= no_prob):
			confidence_pct = yes_prob * 100.0
			st.error(f"YES — Brain Tumor detected with {confidence_pct:.2f}% confidence")
			show_cam = True
			cam_img = generate_gradcam(model, image, target_class_index=yes_idx, imgsz=FIXED_IMGSZ)
		elif no_prob is not None:
			confidence_pct = no_prob * 100.0
			st.success(f"NO — Brain Tumor not detected with {confidence_pct:.2f}% confidence")
		else:
			pred_idx = int(np.argmax(scores))
			pred_prob = scores[pred_idx] * 100.0
			if pred_label.lower() in ("yes", "tumor", "positive"):
				st.error(f"YES — Brain Tumor detected with {pred_prob:.2f}% confidence")
				show_cam = True
				cam_img = generate_gradcam(model, image, target_class_index=pred_idx, imgsz=FIXED_IMGSZ)
			else:
				st.success(f"NO — Brain Tumor not detected with {pred_prob:.2f}% confidence")

		if show_cam and cam_img is not None:
			st.markdown("\n")
			st.image(cam_img, caption="Highlighted tumor region (Grad-CAM)", width=DISPLAY_WIDTH)
	elif uploaded is not None and model is None:
		st.info("Model is not loaded. Train first to create weights at runs/classify/brain_tumor/weights/best.pt.")

st.markdown("---")
st.caption("If detected as YES, a Grad-CAM heatmap highlights influential regions. Images are resized internally to 224x224 for inference.")

import os
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from paddleocr import PaddleOCR, draw_ocr

# Load model and processor
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
ocr = PaddleOCR(det_model_dir="./model/PaddleOCR/ch_PP-OCRv4_det_infer/",
                cls_model_dir="./model/PaddleOCR/cls/",
                rec_model_dir="./model/PaddleOCR/ch_PP-OCRv4_rec_infer",
                use_angle_cls=True,
                lang="ch")

# Load image
path = "images/road.jpg"
result_path = "./images/result/"
image = Image.open(path)
texts = ["Word"]

# OCR detection
ocr_result = ocr.ocr(path, cls=True)
boxes = [line[0] for line in ocr_result[0]]
txts = [line[1][0] for line in ocr_result[0]]
scores = [line[1][1] for line in ocr_result[0]]
im_show = draw_ocr(image, boxes, font_path='./ppocr_img/fonts/simfang.ttf')
ocr_image = Image.fromarray(im_show)

# Process inputs
inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

# Model inference
with torch.no_grad():
    outputs = model(**inputs)

# Get logits and process
logits = outputs.logits
if len(texts) == 1:
    logits = logits.unsqueeze(0)
logits = logits.unsqueeze(1)

# Resize generated image to match the original image size
original_size = (image.height, image.width)
logits_resized = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)

# Create image display
fig, ax = plt.subplots(1, len(texts) + 2, figsize=(3 * (len(texts) + 2), 10))
[a.axis('off') for a in ax]  # Turn off axes for all subplots
ax[0].imshow(image)
# ax[0].set_title("Original Image")
for i in range(len(texts)):
    ax[i + 1].imshow(image)
    ax[i + 1].imshow(torch.sigmoid(logits_resized[i][0]), alpha=0.8)
    # ax[i + 1].set_title(texts[i])
ax[-1].imshow(ocr_image)
# ax[-1].set_title("OCR Results")

# Save the figure without borders
plt.tight_layout(pad=1)  # Adjust layout to prevent clipping

filename = os.path.splitext(os.path.basename(path))[0]
fig.savefig(os.path.join(result_path, filename + '-' + texts[0] + '.png'), bbox_inches='tight', pad_inches=0, dpi=300)
# 显示图像
plt.show()
# Close the figure to free memory
plt.close(fig)

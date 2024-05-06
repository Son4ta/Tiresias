from paddleocr import PaddleOCR, draw_ocr


ocr = PaddleOCR(det_model_dir="./ch_PP-OCRv4_det_infer/",
                cls_model_dir="./cls/",
                rec_model_dir="./ch_PP-OCRv4_rec_infer",
                use_angle_cls=True,
                lang="ch")
img_path = '../../images/peanut.png'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line[1][0])

# 显示结果
from PIL import Image

result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./ppocr_img/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('peanut.jpg')

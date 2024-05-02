import torch
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    get_model_name_from_path,
)
import time
import requests
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from translate import Translator

device = "cuda" if torch.cuda.is_available() else "cpu"
# 这个jb模块老是卡！
def test_translator():
    try:
        Translator(from_lang="English", to_lang="Chinese").translate("mon")
    except:
        raise RuntimeError("网络异常，无法启动翻译")


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files, size=(0, 0)):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        # default no resize
        if size[0] != 0:
            image = image.resize(size)
        out.append(image)
    return out


from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from TTS.api import TTS
import re
from paddleocr import PaddleOCR
from PIL import Image


# translate transcribe
def speech_recognition(audio_file, task="transcribe"):
    # load audio file
    audio, sampling_rate = librosa.load(audio_file, sr=16_000)

    # Load the Whisper model in Hugging Face format:
    processor = WhisperProcessor.from_pretrained("model/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("model/whisper-small",
                                                            low_cpu_mem_usage=True)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task=task)

    # Use the model and processor to transcribe the audio:
    input_features = processor(
        audio, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features

    # Generate token ids
    predicted_ids = model.generate(input_features)

    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print("语音识别结果:{}".format(transcription[0]))
    return transcription[0]


def text_to_speech(prompt, speaker_wav="audio/hanser.wav", language="zh-cn", output_file="output.wav"):
    # 初始化TTS，传入模型名称
    tts = TTS(
        model_path="./model/XTTS-v2",
        config_path="./model/XTTS-v2/config.json",
        progress_bar=False,
    ).to(device)
    # tts = TTS(
        # model_path="./model/zh-CN-tacotron2-DDC-GST/model_file.pth",
        # config_path="./model/zh-CN-tacotron2-DDC-GST/config.json",
        # progress_bar=False).to(device)
    # .to("cuda" if torch.cuda.is_available() else "cpu")
    # 运行TTS，必须设置语言
    # _, wav = tts.tts_to_file(
    #     text=prompt,
    #     speaker_wav=speaker_wav,
    #     language=language,
    #     file_path=output_file,
    # )
    # _, wav = tts.tts_to_file(
    #     text=prompt,
    #     file_path=output_file,
    # )
    if speaker_wav is None:
        _, wav = tts.tts_to_file(
            speaker="Henriette Usha",  # Henriette Usha
            text=prompt,
            language=language,
            file_path=output_file,
            split_sentences=True
        )
    else:
        _, wav = tts.tts_to_file(
            speaker_wav=speaker_wav,  # Henriette Usha
            text=prompt,
            language=language,
            file_path=output_file,
            split_sentences=True
        )
    return wav


def str_preprocess(str1):
    # 字符串预处理byd 小写 去掉该死的“” 加句号 这个模型不支持！
    str1 = re.sub('[“”]', '', str1).lower()
    if not str1.endswith("。"):
        str1 += "。"
    return str1


def ocr(image_file, prompt="OCR result:"):
    # OCR模型调用
    paddle_ocr = PaddleOCR(det_model_dir="./model/PaddleOCR/ch_PP-OCRv4_det_infer/",
                           cls_model_dir="./model/PaddleOCR/cls/",
                           rec_model_dir="./model/PaddleOCR/ch_PP-OCRv4_rec_infer",
                           use_angle_cls=True,
                           lang="ch")
    # OCR识别
    ocr_result = paddle_ocr.ocr(image_file, cls=True)
    # 把结果存result里
    result = []
    for idx in range(len(ocr_result)):
        res = ocr_result[idx]
        for line in res:
            print(line[1][0])
            result.append(line[1][0])
    # 拼接字符串 与prompt结合 prompt是告诉模型从这里开始是OCR结果
    result = prompt + " ".join(result) + '.'
    return result


class Tiresias:
    # LLaVA模型载入
    model_path = ""  # llava-v1.6-mistral-7b llava-v1.5-7b
    model_name = None
    tokenizer, model, image_processor, context_len = None, None, None, None

    def initialize(self):
        # LLaVA模型载入
        self.model_path = "model/llava-v1.5-7b"  # llava-v1.6-mistral-7b llava-v1.5-7b
        disable_torch_init()
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, model_base=None, model_name=self.model_name)

    def caption_image(self, image_file, query, conv_mode="llava_v1", max_new_tokens=512, size=(0, 0)):
        # 处理查询
        qs = query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        # 选择对话模板
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # 加载 预处理 图像
        images = load_images(image_file.split(","), size)
        image_sizes = [x.size for x in images]
        print("图片尺寸:{}".format(image_sizes))
        image_tensor = self.image_processor(images, return_tensors='pt')['pixel_values'].cuda()
        # 将图像 token 添加到输入 token 中
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        # 使用 `torch.inference_mode()` 上下文，以减少内存使用
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return images, outputs


from translate import Translator
import time

query = 'It\'s April 25th, 2025. Is this still drinkable?Why?'
image_file = f'images/coconut.jpg'
audio_file = f'audio/花生.mp3'

# TODO:计时点
start_time = time.time()
test_translator()
Tiresias = Tiresias()
Tiresias.initialize()

# TODO:计时点
SR_process_time = time.time()
ocr_result = ocr(image_file)
query = (#speech_recognition(audio_file, task="translate")
          query
         + Translator(from_lang="zh", to_lang="en").translate(ocr_result)
         + "Short answer."
         )

# TODO:计时点
LLM_process_time = time.time()
image, output = Tiresias.caption_image(image_file, query, size=(640, 640))
print(output)
output_zh = Translator(from_lang="en", to_lang="zh").translate(output)

# TODO:计时点
TTS_process_time = time.time()
# 字符串预处理 加句号
output_zh = str_preprocess(output_zh)
text_to_speech(output_zh, output_file="peanut.wav")

end_time = time.time()
print("初始化耗时: {:.2f}秒".format(SR_process_time - start_time))
print("语音识别推理耗时: {:.2f}秒".format(LLM_process_time - SR_process_time))
print("回答推理耗时: {:.2f}秒".format(TTS_process_time - LLM_process_time))
print("TTS推理耗时: {:.2f}秒".format(end_time - TTS_process_time))

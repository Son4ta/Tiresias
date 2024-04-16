import re

output_zh = "的信息，包括品牌名称“thermos”和型号“jni-5000 dpl-v0032c”。"
output_zh = re.sub('[“”]', '', output_zh)
print(output_zh)

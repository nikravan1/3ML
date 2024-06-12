
### MultiModal MultiLingual (3ML)

This model is 4bit quantized of [glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b) Model (Less than 9G). 

 It excels in document, image, chart questioning answering and delivers superior performance over GPT-4-turbo-2024-04-09, Gemini 1.0 Pro, Qwen-VL-Max, and Claude 3 Opus.

Some part of the original Model changed and It can excute on free version of google colab.
# Try it: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aZGX9f5Yw1WbiOrS3TpvPk_UJUP_yYQU?usp=sharing)

Note: For optimal performance with document and image understanding, please use English or Chinese. The model can still handle chat in any supported language.
### About GLM-4V-9B

GLM-4V-9B is a multimodal language model with visual understanding capabilities. The evaluation results of its related classic tasks are as follows:


|                         | **MMBench-EN-Test** | **MMBench-CN-Test** | **SEEDBench_IMG** | **MMStar** | **MMMU** | **MME** | **HallusionBench** | **AI2D** | **OCRBench** |
|-------------------------|---------------------|---------------------|-------------------|------------|----------|---------|--------------------|----------|--------------|
|                         | 英文综合                | 中文综合                | 综合能力              | 综合能力       | 学科综合     | 感知推理    | 幻觉性                | 图表理解     | 文字识别         |
| **GPT-4o, 20240513**    | 83.4                | 82.1                | 77.1              | 63.9       | 69.2     | 2310.3  | 55                 | 84.6     | 736          |
| **GPT-4v, 20240409**    | 81                  | 80.2                | 73                | 56         | 61.7     | 2070.2  | 43.9               | 78.6     | 656          |
| **GPT-4v, 20231106**    | 77                  | 74.4                | 72.3              | 49.7       | 53.8     | 1771.5  | 46.5               | 75.9     | 516          |
| **InternVL-Chat-V1.5**  | 82.3                | 80.7                | 75.2              | 57.1       | 46.8     | 2189.6  | 47.4               | 80.6     | 720          |
| **LlaVA-Next-Yi-34B**   | 81.1                | 79                  | 75.7              | 51.6       | 48.8     | 2050.2  | 34.8               | 78.9     | 574          |
| **Step-1V**             | 80.7                | 79.9                | 70.3              | 50         | 49.9     | 2206.4  | 48.4               | 79.2     | 625          |
| **MiniCPM-Llama3-V2.5** | 77.6                | 73.8                | 72.3              | 51.8       | 45.8     | 2024.6  | 42.4               | 78.4     | 725          |
| **Qwen-VL-Max**         | 77.6                | 75.7                | 72.7              | 49.5       | 52       | 2281.7  | 41.2               | 75.7     | 684          |
| **GeminiProVision**     | 73.6                | 74.3                | 70.7              | 38.6       | 49       | 2148.9  | 45.7               | 72.9     | 680          |
| **Claude-3V Opus**      | 63.3                | 59.2                | 64                | 45.7       | 54.9     | 1586.8  | 37.8               | 70.6     | 694          |
| **GLM-4v-9B**           | 81.1                | 79.4                | 76.8              | 58.7       | 47.2     | 2163.8  | 46.6               | 81.1     | 786          |
**This repository is the model repository of 4bit quantized of GLM-4V-9B model, supporting `8K` context length.**
## Quick Start

Use colab model or this python script.
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

device = "cuda"

modelPath="nikravan/glm-4vq"
tokenizer = AutoTokenizer.from_pretrained(modelPath, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    modelPath,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
)



query ='explain all the details in this picture'
image = Image.open("a3.png").convert('RGB')
#image=""
inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)  # chat with image mode

inputs = inputs.to(device)

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))
```

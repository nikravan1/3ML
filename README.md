
# MultiModal MultiLingual (3ML) ChatBot

This model is [4bit quantized](https://huggingface.co/nikravan/glm-4vq) of lm-4v-9b Model (Less than 9G). 

 It excels in document, image, chart questioning answering and delivers superior performance over GPT-4-turbo-2024-04-09, Gemini 1.0 Pro, Qwen-VL-Max, and Claude 3 Opus.

Some part of the original Model changed and It can excute on free version of google colab.
### Try it with gradio support:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VH1tlpl_N4nRS2q5k0lxu5H_taEL0MOw?usp=sharing)

Note: For optimal performance with document and image understanding, please use English or Chinese docs. The model can still handle chat in any supported language.

## Quick Start


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
##Samples:
![Screenshot from 2024-06-12 17-52-59](https://github.com/nikravan1/3ML/assets/17721448/d9a27314-d539-471c-9a26-3cf98890a8e5)

![Screenshot from 2024-06-12 17-54-59](https://github.com/nikravan1/3ML/assets/17721448/7cf77373-1d3f-4d01-a64a-0eb7478309f7)

![image](https://github.com/nikravan1/3ML/assets/17721448/3aeca087-ebb2-47cc-b357-6331c6470b67)
![Screenshot from 2024-06-12 18-01-05](https://github.com/nikravan1/3ML/assets/17721448/bdfe943b-0b0a-4477-b822-38726b67f44d)







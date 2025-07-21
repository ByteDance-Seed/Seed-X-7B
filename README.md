<div align="center">
 👋 Hi, everyone! 
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://seed.bytedance.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/5793e67c-79bb-4a59-811a-fcc7ed510bd4">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

<div align=center>
<img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216"/></div>
</div>

<!-- 注释：以上为Seed官方信息，可直接复制使用，请注意导入“Seed WeChat”（第12行）、“Seed logo”(第20行)图片替换 -->


# Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters
<p align="center">
  <a href="https://arxiv.org/pdf/2507.13618">
    <img src="https://img.shields.io/badge/Seed--X-Report-blue"></a>
  <a href="https://huggingface.co/collections/ByteDance-Seed/seed-x-6878753f2858bc17afa78543">
    <img src="https://img.shields.io/badge/Seed--X-Hugging Face-brightgreen"></a>
  <a href="https://github.com/ByteDance-Seed/Seed-X-7B/blob/main/LICENSE.openmdw">
    <img src="https://img.shields.io/badge/License-OpenMDW-yellow"></a>
</p>

<!-- 🤗 [HuggingFace]() | 📄 [Technical Report](/Technical_Report.pdf) -->


We are excited to introduce **Seed-X**, a powerful series of open-source multilingual translation language models, including an instruction model, a reinforcement learning model, and a reward model. It pushes the boundaries of translation capabilities within 7 billion parameters.

<!-- 注释：以上为项目基础信息，以项目COMET举例，Comet一级标题（第25行）、徽章Comet名字（第28、30、32、34行）记得替换，徽章可按需使用
请注意，徽章可根据具体项目自定义，如技术成果落地页、技术成果报告/Paper、Hugging Face、项目微信交流群、License、打榜榜单等，更换名字和链接即可；
专属微信群出现在两个位置，第34行、第42行，可以联系EB同学创建 -->

## 📢 News
[2025/07/18] 🔥 We have released the Seed-X-Challenge-Set.
<br>
[2025/07/18] 🔥 Seed-X-Instruct/PPO/RM are now avaliable on Huggingface!

## 🌟 Highlights

* **Exceptional translation capabilities**: Seed-X exhibits state-of-the-art translation capabilities, on par with or outperforming ultra-large models like Gemini-2.5, Claude-3.5, and GPT-4, as validated by human evaluations and automatic metrics.
* **Deployment and inference-friendly**: With a compact 7B parameter count and mistral architecture, Seed-X offers outstanding translation performance in a lightweight and efficient package, ideal for deployment and inference.
* **Broad domain coverage**: Seed-X excels on a highly challenging translation test set spanning diverse domains, including the internet, science and technology, office dialogues, e-commerce, biomedicine, finance, law, literature, and entertainment.

## 🏆 Performance

We evaluated Seed-X on a diverse set of translation benchmarks, including FLORES-200, WMT-25, and a publicly released [challenge set](https://github.com/ByteDance-Seed/Seed-X-7B/tree/main/challenge_set) accompanied by human evaluations.

![performance](/imgs/model_comparsion.png)

For detailed benchmark results and analysis, please refer to our [Technical Report](https://arxiv.org/pdf/2507.13618).

## ⚡ Quick Start
We are excited to introduce Seed-X, featuring three powerful models:

| Model Name  | Description | Download |
| ----------- | ----------- |-----------
| Seed-X-Instruct  | Instruction-tuned for alignment with user intent. |🤗 [Model](https://huggingface.co/ByteDance-Seed/Seed-X-Instruct-7B)|
| Seed-X-PPO | RL trained to boost translation capabilities.     | 🤗 [Model](https://huggingface.co/ByteDance-Seed/Seed-X-PPO-7B)|
|Seed-X-RM | Reward model to evaluate the quality of translation.|  🤗 [Model](https://huggingface.co/ByteDance-Seed/Seed-X-RM-7B)| 

### 👉 Deploying Seed-X-PPO with ```vllm```

❗**The language tags at the end of the prompt is necessary**, which are used in PPO training. For example, when the target language is German, \<de\> needs to be added. You can refer to the above table for language abbreviations.

❗**This model is specialized in multilingual translation**, which is unexpected to support other tasks. 

❗**We don't have any chat template**, thus you don't have to perform ```tokenizer.apply_chat_template```. Please avoid prompting the model in a multi-round conversation format.

❗**We recommend against using unofficial quantized versions for local deployment.** We will soon release an official quantized model and develop a demo on Hugging Face Space.

Here is a simple example demonstrating how to load the model and perform translation using ```vllm```
```python
from vllm import LLM, SamplingParams, BeamSearchParams

model_path = "./ByteDance-Seed/Seed-X-PPO-7B"

model = LLM(model=model_path,
            max_num_seqs=512,
            tensor_parallel_size=8,
            enable_prefix_caching=True, 
            gpu_memory_utilization=0.95)

messages = [
    "Translate the following English sentence into Chinese:\nMay the force be with you <zh>", # without CoT
    "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>" # with CoT
]

# Beam search (We recommend using beam search decoding)
decoding_params = BeamSearchParams(beam_width=4,
                                   max_tokens=512)
# Greedy decoding
decoding_params = SamplingParams(temperature=0,
                                 max_tokens=512,
                                 skip_special_tokens=True)

results = model.generate(messages, decoding_params)
responses = [res.outputs[0].text.strip() for res in results]

print(responses)
```
## License
This project is licensed under OpenMDW. See the [LICENSE](https://github.com/ByteDance-Seed/Seed-X-7B/blob/main/LICENSE.openmdw) file for details.

## Citation
```bibtex
@misc{cheng2025seedxbuildingstrongmultilingual,
      title={Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters}, 
      author={Shanbo Cheng and Yu Bao and Qian Cao and Luyang Huang and Liyan Kang and Zhicheng Liu and Yu Lu and Wenhao Zhu and Jingwen Chen and Zhichao Huang and Tao Li and Yifu Li and Huiying Lin and Sitong Liu and Ningxin Peng and Shuaijie She and Lu Xu and Nuo Xu and Sen Yang and Runsheng Yu and Yiming Yu and Liehao Zou and Hang Li and Lu Lu and Yuxuan Wang and Yonghui Wu},
      year={2025},
      eprint={2507.13618},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.13618}, 
}
```
## About [ByteDance Seed Team](https://seed.bytedance.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.

<!-- 注释：About ByteDance Seed Team可直接复制使用 -->


# ****List of Open-Source Finetuned Large Language Models****

This repository contains a curated (incomplete) list of open-source and finetuned Large Language Models. 

![Lama](https://images.unsplash.com/photo-1661345440932-85b638c8088f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1771&q=80)

**************************************************************************************************

****LLaMA (Meta)****
>LLaMA (Large Language Model Meta AI), a state-of-the-art foundational large language model designed to help researchers advance their work in this subfield of AI. Smaller, more performant models such as LLaMA enable others in the research community who don’t have access to large amounts of infrastructure to study these models, further democratizing access in this important, fast-changing field.

- LLaMA Website: [Introducing LLaMA: A foundational, 65-billion-parameter language model (facebook.com)](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)

**Alpaca (Stanford)**
>We are releasing our findings about an instruction-following language model, dubbed Alpaca, which is fine-tuned from Meta’s LLaMA 7B model. We train the Alpaca model on 52K instruction-following demonstrations generated in the style of self-instruct using text-davinci-003. On the self-instruct evaluation set, Alpaca shows many behaviors similar to OpenAI’s text-davinci-003, but is also surprisingly small and easy/cheap to reproduce.

- Website: [https://crfm.stanford.edu/2023/03/13/alpaca.html](https://crfm.stanford.edu/2023/03/13/alpaca.html)

- GitHub: [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)

****Alpaca-LoRA****
>This repository contains code for reproducing the Stanford Alpaca results using low-rank adaptation (LoRA). We provide an Instruct model of similar quality to text-davinci-003 that can run on a Raspberry Pi (for research), and the code is easily extended to the 13b, 30b, and 65b models.

- GitHub: [tloen/alpaca-lora: Instruct-tune LLaMA on consumer hardware (github.com)](https://github.com/tloen/alpaca-lora)

- Demo: [Alpaca-LoRA — a Hugging Face Space by tloen](https://huggingface.co/spaces/tloen/alpaca-lora)

****Baize****

- GitHub: [project-baize/baize: Baize is an open-source chatbot trained with ChatGPT self-chatting data, developed by researchers at UCSD and Sun Yat-sen University. (github.com)](https://github.com/project-baize/baize)

- Paper: [2304.01196.pdf (arxiv.org)](https://arxiv.org/pdf/2304.01196.pdf)

****Koala****

- GitHub: [EasyLM/koala.md at main · young-geng/EasyLM (github.com)](https://github.com/young-geng/EasyLM/blob/main/docs/koala.md)

- Blog: [Koala: A Dialogue Model for Academic Research — The Berkeley Artificial Intelligence Research Blog](https://bair.berkeley.edu/blog/2023/04/03/koala/)

****Vicuna (FastChat)****
> We introduce Vicuna-13B, an open-source chatbot trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT. Preliminary evaluation using GPT-4 as a judge shows Vicuna-13B achieves more than 90%* quality of OpenAI ChatGPT and Google Bard while outperforming other models like LLaMA and Stanford Alpaca in more than 90%* of cases. The cost of training Vicuna-13B is around $300. The code and weights, along with an online demo, are publicly available for non-commercial use.
- GitHub: [lm-sys/FastChat: The release repo for “Vicuna: An Open Chatbot Impressing GPT-4” (github.com)](https://github.com/lm-sys/FastChat)
- Website: [Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality](https://lmsys.org/blog/2023-03-30-vicuna/)

****llama.cpp****
> LLama.cpp, allows users to run the LLaMA model on their local computers using C/C++. According to the documentation, llama.cpp supports the following models and runs on moderately speed PCs:
> 
> LLaMA | Alpaca | GPT4All | Vicuna | Koala | OpenBuddy (Multilingual) | Pygmalion 7B / Metharme 7B

- GitHub: [ggerganov/llama.cpp: Port of Facebook’s LLaMA model in C/C++ (github.com)](https://github.com/ggerganov/llama.cpp)


****LLaMA-Adapter V2****

- GitHub: [ZrrSkywalker/LLaMA-Adapter: Fine-tuning LLaMA to follow Instructions within 1 Hour and 1.2M Parameters (github.com)](https://github.com/ZrrSkywalker/LLaMA-Adapter)

****Lit-LLaMA ️****

- GitHub: [Lightning-AI/lit-llama: Implementation of the LLaMA language model based on nanoGPT. Supports quantization, LoRA fine-tuning, pre-training. Apache 2.0-licensed. (github.com)](https://github.com/Lightning-AI/lit-llama)


****StableVicuna****

- Website: [Stability AI releases StableVicuna, the AI World’s First Open Source RLHF LLM Chatbot — Stability AI](https://stability.ai/blog/stablevicuna-open-source-rlhf-chatbot)

- Hugging Face: [StableVicuna — a Hugging Face Space by CarperAI](https://huggingface.co/spaces/CarperAI/StableVicuna)

****StackLLaMA****

- Website: [https://huggingface.co/blog/stackllama](https://huggingface.co/blog/stackllama)

****StableLM (StabilityAI)****

- Website: [Stability AI Launches the First of its StableLM Suite of Language Models — Stability AI](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models)
- GitHub: [Stability-AI/StableLM: StableLM: Stability AI Language Models (github.com)](https://github.com/stability-AI/stableLM/)
- Hugging Face: [Stablelm Tuned Alpha Chat — a Hugging Face Space by stabilityai](https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat)

****GPT4All****
>GTP4All is an ecosystem to train and deploy powerful and customized large language models that run locally on consumer grade CPUs.
The goal is simple - be the best instruction tuned assistant-style language model that any person or enterprise can freely use, distribute and build on.
>
- GitHub: [nomic-ai/gpt4all: gpt4all: a chatbot trained on a massive collection of clean assistant data including code, stories and dialogue (github.com)](https://github.com/nomic-ai/gpt4all)

- GitHub: [nomic-ai/pyllamacpp: Official supported Python bindings for llama.cpp + gpt4all (github.com)](https://github.com/nomic-ai/pyllamacpp)

****GPT-J (EleutherAI)****

- GitHub: [https://github.com/kingoflolz/mesh-transformer-jax/#gpt-j-6b](https://github.com/kingoflolz/mesh-transformer-jax/#gpt-j-6b)

****GPT4All-J****

- GitHub: [nomic-ai/gpt4all: gpt4all: an ecosystem of open-source chatbots trained on a massive collections of clean assistant data including code, stories and dialogue (github.com)](https://github.com/nomic-ai/gpt4all)

****GPT-NeoX (EleutherAI)****

- GitHub: [EleutherAI/gpt-neox: An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library. (github.com)](https://github.com/EleutherAI/gpt-neox)

****Pythia (EleutherAI)****

- GitHub: [EleutherAI/pythia (github.com)](https://github.com/EleutherAI/pythia)

****Dolly 2.0 (Databricks)****
> Databricks’ Dolly is an instruction-following large language model trained on the Databricks machine learning platform that is licensed for commercial use. Based on pythia-12b, Dolly is trained on ~15k instruction/response fine tuning records databricks-dolly-15k generated by Databricks employees in capability domains from the InstructGPT paper, including brainstorming, classification, closed QA, generation, information extraction, open QA and summarization.
- GutHub: [dolly/data at master · databrickslabs/dolly (github.com)](https://github.com/databrickslabs/dolly/tree/master/data)
- Blog post: [Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- Hugging Face: [databricks (Databricks) (huggingface.co)](https://huggingface.co/databricks)

****OpenAssistant Models****
> Open Assistant is a project meant to give everyone access to a great chat based large language model.
We believe that by doing this we will create a revolution in innovation in language. In the same way that stable-diffusion helped the world make art and images in new ways we hope Open Assistant can help improve the world by improving language itself.

- GitHub: [LAION-AI/Open-Assistant: OpenAssistant is a chat-based assistant that understands tasks, can interact with third-party systems, and retrieve information dynamically to do so.](https://github.com/LAION-AI/Open-Assistant)
- Website: [Open Assistant](https://open-assistant.io/chat)

****Replit-Code (Replit)****

- Hugging Face: [https://huggingface.co/replit/replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b)

****Segment Anything (Meta)****
> We aim to democratize segmentation by introducing the Segment Anything project: a new task, dataset, and model for image segmentation, as we explain in our research paper. We are releasing both our general Segment Anything Model (SAM) and our Segment Anything 1-Billion mask dataset (SA-1B), the largest ever segmentation dataset, to enable a broad set of applications and foster further research into foundation models for computer vision.
- GitHub: [facebookresearch/segment-anything: The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model. (github.com)](https://github.com/facebookresearch/segment-anything)
- Website: [Segment Anything](https://segment-anything.com/)

****StartCoder (BigCode)****

- Website: [https://huggingface.co/bigcode](https://huggingface.co/bigcode)
- Hugging Face: [https://huggingface.co/spaces/bigcode/bigcode-editor](https://huggingface.co/spaces/bigcode/bigcode-editor) and [https://huggingface.co/spaces/bigcode/bigcode-playground](https://huggingface.co/spaces/bigcode/bigcode-playground)

****BLOOM (BigScience)****

- Hugging Face: [bigscience/bloom · Hugging Face](https://huggingface.co/bigscience/bloom)

****Flamingo (Google/Deepmind)****

- Website: [Tackling multiple tasks with a single visual language model](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)

- GitHub: [https://github.com/lucidrains/flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch)

****FLAN (Google)****

- GitHub: [google-research/FLAN (github.com)](https://github.com/google-research/FLAN)

****FastChat-T5****

- GitHub: [lm-sys/FastChat: The release repo for “Vicuna: An Open Chatbot Impressing GPT-4” (github.com)](https://github.com/lm-sys/FastChat#FastChat-T5)

****Flan-Alpaca****

- GitHub: [declare-lab/flan-alpaca: This repository contains code for extending the Stanford Alpaca synthetic instruction tuning to existing instruction-tuned models such as Flan-T5. (github.com)](https://github.com/declare-lab/flan-alpaca)

************************************************************************************************************

**Commercial Use LLMs**

Pythia | Dolly |Open Assistant (Pythia family) | GPT-J-6B | GPT-NeoX | 

Bloom | StableLM-Alpha | FastChat-T5 |

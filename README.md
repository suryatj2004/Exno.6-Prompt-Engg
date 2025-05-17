# Exno.6-Prompt-Engg
## Date:
## Register no: 212222040168
## Aim: 
Development of Python Code Compatible with Multiple AI Tools

## Algorithm: 
Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights.

## Objective:
Build a Python script that:

a. Connects to multiple AI services via APIs.

b. Sends standardized prompts or inputs.

c. Collects and stores the outputs.

d. Compares the outputs for quality, tone, performance, or accuracy.

e. Generates reports or logs for further analysis.

This process helps in benchmarking AI tools and determining the best tool for a particular task or use case.

## Procedure / Algorithm:
### Step 1: Define the Use Case
#### AI Tools by Modality
### 1. Image Generation
Stability AI (Stable Diffusion via Stability API)

OpenAI DALL·E

Hugging Face Diffusers (local or hosted models)

### 2. Voice Enhancement
Adobe Enhance Speech (API access via Adobe Podcast)

ElevenLabs (for voice synthesis and enhancement)

iZotope RX (for pro audio cleanup, often used offline)

### 3. Video Generation
Runway ML (Gen-2)

Pika Labs

Synthesia (for avatar videos, more templated)

## 1. Image Generation (Stable Diffusion via Stability AI)
```

from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt: str, output_path="output_image.png"):
    print("Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        revision="fp16"
    ).to("cuda")

    print(f"Generating image for prompt: {prompt}")
    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"✅ Image saved to {output_path}")

# Example
generate_image("A futuristic city under the stars", "city_image.png")
```
![image](https://github.com/user-attachments/assets/d8faf2a0-f067-4bce-9e5f-4bfc4aa0891e)

## 2. VOICE ENHANCEMENT (Noisereduce)
```
import noisereduce as nr
import librosa
import soundfile as sf

def enhance_audio(input_path: str, output_path: str):
    print(f"Loading audio file: {input_path}")
    y, sr = librosa.load(input_path, sr=None)
    print("Reducing noise...")
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_path, reduced_noise, sr)
    print(f"✅ Enhanced audio saved to {output_path}")

# Upload an audio file to Colab for testing
from google.colab import files
print("⬆️ Upload a noisy WAV file (mono, short length recommended):")
uploaded = files.upload()

for fname in uploaded:
    enhance_audio(fname, "enhanced_" + fname)
```
![image](https://github.com/user-attachments/assets/2be934da-4bfe-4edf-9fbe-ebea2db5d8ce)

## 3. VIDEO GENERATION (ModelScope)
```
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import os

def generate_video(prompt: str):
    print("Loading ModelScope video generator...")
    text2video = pipeline(task="text-to-video-synthesis", model="damo/text-to-video-synthesis")
    print(f"Generating video for prompt: {prompt}")
    result = text2video({'text': prompt})
    video_path = result[OutputKeys.OUTPUT_VIDEO]
    print(f"✅ Video saved to: {video_path}")
    return video_path

# Example
video_file = generate_video("A panda riding a bicycle in Times Square")

# Download link
from IPython.display import HTML
HTML(f'<video width=400 controls><source src="{video_file}" type="video/mp4"></video>')
```
![image](https://github.com/user-attachments/assets/d4bd9227-f0be-41df-81ab-3af84b960977)


## Conclusion:
This experiment demonstrates how Python can serve as a powerful bridge between multiple AI tools, enabling developers to create multi-model pipelines that evaluate, compare, or combine the strengths of various services. This integration supports:

1. Better decision-making on tool selection

2. Automation of evaluation and benchmarking

3. Enhanced productivity by combining outputs

Such a system is scalable and can be adapted for broader use cases including multi-tool chatbots, creative content workflows, or research benchmarking.


## Result: 
The corresponding Prompt is executed successfully

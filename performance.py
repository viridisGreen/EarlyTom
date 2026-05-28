import os
import time
import torch
import warnings
import numpy as np
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from holitom import holitom
from thop import profile, clever_format
import copy

warnings.filterwarnings("ignore")

# ======================
# 1. Config
# ======================
VIDEO_PATH = "../EarlyTom/LLaVA-NeXT/docs/jobs.mp4"
MAX_FRAMES = 32
BATCH_SIZE = 1
WARMUP = 2
REPEATS = 5

pretrained = "/path/to/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

# ======================
# 2. Load Model
# ======================
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name,
    device_map=device_map,
    attn_implementation="sdpa",
    multimodal=True,
)

model_wrapper = os.environ.get("WRAPPER")
if model_wrapper in ["visionzip", "holitom", "earlytom"]:
    wrapper_module = __import__(model_wrapper)
    wrapper_class = getattr(wrapper_module, model_wrapper)
    model = wrapper_class(model)
elif model_wrapper is not None:
    print("Vanilla")
model.eval()

# ======================
# 3. Load Video
# ======================
def load_video(video_path, max_frames_num=32):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    frame_idx = np.linspace(0, total - 1, max_frames_num, dtype=int).tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames  # (T, H, W, 3)

video_frames = load_video(VIDEO_PATH, MAX_FRAMES)
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda(device)
image_tensors = [frames]
image_sizes = [frame.size for frame in video_frames]

# ======================
# 4. Prepare Input
# ======================
conv_template = "qwen_1_5"
conv = copy.deepcopy(conv_templates[conv_template])
question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

# ======================
# 5. Measure Decoding Throughput
# ======================
print("\n==== Measuring Decoding Throughput (Tokens/s) ====")
times, tokens = [], []

for i in range(WARMUP + REPEATS):
    torch.cuda.synchronize()
    # start = time.time()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event =torch.cuda.Event(enable_timing=True)
    start_event.record()
    outputs = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )
    end_event.record()
    torch.cuda.synchronize()

    if i >= WARMUP:
        elapsed = start_event.elapsed_time(end_event)/1000
        gen_tokens = outputs.shape[1] - input_ids.shape[1]
        times.append(elapsed)
        tokens.append(gen_tokens)

avg_time = np.mean(times)
avg_tokens = np.mean(tokens)
tps = avg_tokens / avg_time
print(f"🕒 Avg decoding time: {avg_time:.3f}s, "
      f"Avg tokens: {avg_tokens:.1f}, "
      f"Throughput: {tps:.2f} tokens/s")


# ======================
# 6 Measure TTFT (Time To First Token)
# ======================
print("\n==== Measuring TTFT (Time to First Token) ====")

ttft_records = []

for i in range(WARMUP + REPEATS):

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    first_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    outputs = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=1,
        modalities=["video"],
        return_dict_in_generate=True,
        output_scores=True,
    )
    first_event.record()
    torch.cuda.synchronize()

    elapsed = start_event.elapsed_time(first_event) / 1000  # ms -> s

    if i >= WARMUP:
        ttft_records.append(elapsed)

avg_ttft = sum(ttft_records) / len(ttft_records)
print(f"✅ Avg TTFT = {avg_ttft:.3f} seconds ({REPEATS} runs, {WARMUP} warmup)")


# ======================
# 7. Measure FLOPs (Full Model)
# ======================
import torch
from torch import nn
from thop import profile, clever_format
from deepspeed.profiling.flops_profiler import get_model_profile

class ProfileWrapper(nn.Module):
    """
    Wraps a multimodal model to make it compatible with thop.profile().
    Only runs a single forward pass with typical arguments.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, images, image_sizes):
        out = self.model(
            input_ids=input_ids,
            images=images,
            image_sizes=image_sizes,
            modalities=["video"],
            output_hidden_states=False,
        )
        return out
    
print("\n==== Measuring Full Model FLOPs ====")

dummy_frames = video_frames
dummy_image_tensors = image_processor.preprocess(dummy_frames, return_tensors="pt")["pixel_values"].half().cuda(device)
dummy_image_sizes = [frame.shape[:2] for frame in dummy_frames]
dummy_input_ids = input_ids[:, :32]

model.eval()
flops, macs, params = get_model_profile(model, 
                                        kwargs=dict(
                                        input_ids=dummy_input_ids,
                                        images=dummy_image_tensors,
                                        image_sizes=dummy_image_sizes,
                                        modalities=["video"],
                                        output_hidden_states=False,
                                    ))

print(f"📊 Full Model FLOPs: {flops}, Params: {params}, MACs: {macs}")

# ======================
# 7. Summary
# ======================
print("\n===== ✅ Performance Summary =====")
print(f"Model: {model_name}")
print(f"Video: {os.path.basename(VIDEO_PATH)} ({MAX_FRAMES} frames)")
print(f"Decoding Throughput: {tps:.2f} tokens/s")
print(f"TTFT: {avg_ttft:.3f} seconds")
print(f"Full Model FLOPs: {flops}")
print(f"MACs: {macs}")
print(f"Params: {params}")
print("=================================")

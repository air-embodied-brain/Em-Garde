import math
from typing import List, Optional, TypeAlias, Union
import types

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs

import time

ImageInput: TypeAlias = Union[Image.Image, List[Image.Image]]
BatchImageInput: TypeAlias = Union[List[Image.Image], List[List[Image.Image]]]


def streaming_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_cached_embeds: Optional[torch.FloatTensor] = None,
        max_image_num: Optional[int] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
                if image_cached_embeds is not None:
                    image_embeds = torch.cat([image_cached_embeds, image_embeds], dim=0)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                tokens_per_image = image_grid_thw[0].prod().item() // 4
                new_image_embeds_cache = image_embeds[tokens_per_image:].detach() if image_embeds.shape[0] // tokens_per_image == max_image_num else image_embeds.detach()
                
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        total_image_grid_thw = image_grid_thw.repeat(image_embeds.shape[0] // image_grid_thw.shape[0], 1)  # assume total number of image tokens is a multiple of the number of new images
        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.model.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids, total_image_grid_thw, video_grid_thw, attention_mask
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.model.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
        return outputs.hidden_states[-1], new_image_embeds_cache
 
class Qwen2VLImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class Qwen2VLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Qwen2VLImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    } 
    
def stream_call(
        self,
        images = None,
        text  = None,
        videos = None,
        **kwargs,
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    # index += 1       # always use the first image_thw in streaming case
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.video_token, "<|placeholder|>" * num_video_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

class OpsMMEmbeddingV1(nn.Module):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: Optional[int] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.default_instruction = "You are a helpful assistant."
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        ).to(self.device)
        
        self.base_model.streaming_forward = types.MethodType(streaming_forward, self.base_model)
        
        self.current_streaming_length = 0
        self.image_cached_embeds = None

        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=4 * 28 * 28, max_pixels=1280 * 28 * 28)
        self.processor.tokenizer.padding_side = "left"
        self.processor.stream_call = types.MethodType(stream_call, self.processor)
        
        self.eval()

    def reset(self):
        self.current_streaming_length = 0
        self.image_cached_embeds = None
    
    def encode_input(self, input):
        start_time = time.time()
        # print("input ids shape ", input['input_ids'].shape)
        # print("video_grid_thw", input['video_grid_thw'] if 'video_grid_thw' in input else "N/A")
        # print("image_grid_thw", input['image_grid_thw'] if 'image_grid_thw' in input else "N/A")
        hidden_states = self.base_model(**input, return_dict=True, output_hidden_states=True)
        end_time = time.time()
        # print(f"Encoder forward time: {end_time - start_time} seconds")
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states)
        return pooled_output

    def _pooling(self, last_hidden_state):
        batch_size = last_hidden_state.shape[0]
        reps = last_hidden_state[torch.arange(batch_size), -1, :]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def _validate_instructions(
        self,
        texts: Optional[List[str]],
        images: Optional[BatchImageInput],
        instruction: Optional[Union[str, List[str]]],
    ) -> List[str]:
        """Validate and format instructions to match batch size"""
        batch_size = max(len(x) if x is not None else 0 for x in [texts, images])

        if instruction is None:
            return [self.default_instruction] * batch_size

        if isinstance(instruction, str):
            return [instruction] * batch_size

        if isinstance(instruction, list):
            if len(instruction) != batch_size:
                raise ValueError(f"Length of instruction list ({len(instruction)}) must match batch size ({batch_size}) when texts/images are provided")
            return instruction

        raise TypeError("instruction must be str, List[str] or None")

    def _process_images(self, images: ImageInput) -> List[Image.Image]:
        """Convert single image or list of images to processed format"""
        if isinstance(images, Image.Image) or isinstance(images, str):
            return [fetch_image(images)]
        return [fetch_image(i) for i in images]

    def embed(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[BatchImageInput] = None,
        instruction: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate embeddings for text, images, or combined inputs.

        Args:
            texts: List of text inputs (optional)
            images: Can be:
                - List[Image.Image]: Single image per input
                - List[List[Image.Image]]: Multiple images per input
            instruction: Instruction(s) for the model. Can be:
                - None: use default instruction
                - str: use same instruction for all inputs
                - List[str]: per-input instructions (must match batch size)
        """
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        instructions = self._validate_instructions(texts, images, instruction)

        # Determine batch size
        batch_size = len(texts) if texts is not None else len(images)  # type: ignore

        input_texts, input_images = [], []
        for i in range(batch_size):
            text = texts[i] if texts is not None else None
            image = images[i] if images is not None else None

            input_str = ""
            processed_image = None
            if image is not None:
                processed_image = self._process_images(image)
                input_str += "<|vision_start|><|image_pad|><|vision_end|>" * len(processed_image)

            if text is not None:
                input_str += text

            msg = f"<|im_start|>system\n{instructions[i]}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"

            input_texts.append(msg)
            input_images.append(processed_image)

        # Only pass to processor if we actually have images
        processed_images = input_images if any(img is not None for img in input_images) else None

        inputs = self.processor(
            text=input_texts,
            images=processed_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # print("image_grid_thw", inputs['image_grid_thw'] if 'image_grid_thw' in inputs else "N/A")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        start_time = time.time()
        with torch.inference_mode():
            embeddings = self.encode_input(inputs)
        end_time = time.time()
        # print("model forward time:", end_time - start_time)

        return embeddings

    def get_text_embeddings(
        self,
        texts: List[str],
        instruction: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Convenience method for text-only embeddings"""
        return self.get_fused_embeddings(texts=texts, instruction=instruction, **kwargs)

    def get_image_embeddings(
        self,
        images: BatchImageInput,
        instruction: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Convenience method for image-only embeddings.

        Args:
            images: Can be:
                - List[Image.Image]: Single image per input
                - List[List[Image.Image]]: Multiple images per input
        """
        return self.get_fused_embeddings(images=images, instruction=instruction, **kwargs)

    def get_fused_embeddings(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[BatchImageInput] = None,
        instruction: Optional[Union[str, List[str]]] = None,
        batch_size: int = 8,
        show_progress: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Batch processing for large collections of texts/images.

        Args:
            texts: List of text inputs (optional)
            images: Can be:
                - List[Image.Image]: Single image per input
                - List[List[Image.Image]]: Multiple images per input
            instruction: Instruction(s) for the model
            batch_size: Number of items to process at once
            show_progress: Whether to display progress bar
        """

        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        total_items = len(texts) if texts is not None else len(images)  # type: ignore
        num_batches = math.ceil(total_items / batch_size)

        all_embeddings = []
        progress = tqdm(total=num_batches, disable=not show_progress, desc="Processing")
        
        start_time = time.time()
        for i in range(0, total_items, batch_size):
            batch_texts = texts[i : i + batch_size] if texts is not None else None
            batch_images = images[i : i + batch_size] if images is not None else None
            batch_emb = self.embed(texts=batch_texts, images=batch_images, instruction=instruction)

            all_embeddings.append(batch_emb.cpu())
            progress.update(1)
        end_time = time.time()
        # print("Total embedding time:", end_time - start_time)

        progress.close()
        return torch.cat(all_embeddings, dim=0).to(self.device)

    def forward(self, **inputs) -> torch.Tensor:
        """Alias for encode_input"""
        return self.encode_input(inputs)
    
    def streaming_embed(self, new_frames, frames_length=10):
        if self.current_streaming_length < frames_length:
            self.current_streaming_length += len(new_frames)
        input_str = "<|vision_start|><|image_pad|><|vision_end|>" * self.current_streaming_length
        instruction = self.default_instruction
        msg = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>\n"
        processed_frames = self._process_images(new_frames)
        inputs = self.processor.stream_call(
            text=[msg],
            images=[processed_frames],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # print(inputs.image_grid_thw)
        # print("image_grid_thw", inputs['image_grid_thw'] if 'image_grid_thw' in inputs else "N/A")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            hidden_states, new_image_embeds_cache = self.base_model.streaming_forward(**inputs, image_cached_embeds = self.image_cached_embeds, max_image_num=frames_length)
        pooled_output = self._pooling(hidden_states)
        self.image_cached_embeds = new_image_embeds_cache
        return pooled_output


### Modified from qwen_vl_utils.vision_process.py
import base64
import logging
import math
from io import BytesIO

import requests

IMAGE_FACTOR = 28
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int | float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int | float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
        logging.warning(f"Absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(h_bar, w_bar) / min(h_bar, w_bar)}")
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO
    return h_bar, w_bar


def fetch_image(
    image: str | Image.Image | torch.Tensor,
    size_factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Image.Image | torch.Tensor:
    image_obj = None
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)  # type: ignore
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))

    return image


###

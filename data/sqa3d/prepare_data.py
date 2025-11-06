import os
import re
import json
import yaml
import torch
import pprint
from collections import defaultdict
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from datasets import load_dataset

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)


CUSTOM_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{%- if tools %}{{- '<|im_start|>system\\n' }}{%- if messages[0]['role'] == 'system' %}{%- if messages[0]['content'] is string %}{{- messages[0]['content'] }}{%- else %}{{- messages[0]['content'][0]['text'] }}{%- endif %}{%- else %}{{- 'You are a helpful assistant.' }}{%- endif %}{{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}{%- for tool in tools %}{{- \"\\n\" }}{{- tool | tojson }}{%- endfor %}{{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}{% for message in messages %}{% if message['role'] != 'system' or loop.first == false %}{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{%- elif message.role == \"assistant\" %}{{- '<|im_start|>' + message.role }}{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- '\\n<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- '<|im_start|>user' }}{%- endif %}{{- '\\n<tool_response>\\n' }}{% if message['content'] is string %}{{ message.content }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'text' or 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{- '\\n</tool_response>' }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}{% endif %}{% endfor %}{%- else %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{%- elif message.role == \"assistant\" %}{{- '<|im_start|>' + message.role }}{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- '\\n<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- '<|im_start|>user' }}{%- endif %}{{- '\\n<tool_response>\\n' }}{% if message['content'] is string %}{{ message.content }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'text' or 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{- '\\n</tool_response>' }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}{% endfor %}{%- endif %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"


def render_chat_with_template(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    add_vision_id: bool = False,
    template_str: str | None = None,
    normalize: bool = False,
) -> str:
    """
    Apply the Qwen-VL Jinja chat template to messages (and optional tools).
    - messages: list of {'role': str, 'content': str|list[segments], ...}
    - tools: OpenAI-style function tool schemas (list of dicts) or None
    - add_generation_prompt: whether to append assistant header at the end
    - add_vision_id: whether to prefix images/videos with 'Picture N:'/'Video N:'
    - template_str: override template string; defaults to CUSTOM_TEMPLATE
    - normalize: if True, converts '<image>' markers in strings to segment lists
    """
    if normalize:
        messages = update_prompt_json_like_build_messages(messages)

    tpl = template_str or CUSTOM_TEMPLATE

    # Local import to avoid hard dependency at module import time
    from jinja2 import Environment, StrictUndefined

    env = Environment(
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    template = env.from_string(tpl)
    return template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        add_vision_id=add_vision_id,
    )

def update_prompt_json_like_build_messages(
    prompt_json: List[Dict[str, Any]],
    image_token: str = "<image>",
    video_token: str = "<video>",
    convert_plain_text: bool = False,
) -> List[Dict[str, Any]]:
    """
    Transform string `content` fields in prompt_json into a list of segments:
    - {'type': 'image'} for each <image>
    - {'type': 'video'} for each <video>
    - {'type': 'text', 'text': '<text>'} for other text

    If `convert_plain_text` is False, only messages containing image/video markers
    are converted; otherwise, any string content becomes a single 'text' segment.

    Returns a new list (does not mutate the input list).
    """
    out = []
    pattern = re.compile(rf"({re.escape(image_token)}|{re.escape(video_token)})")

    for msg in prompt_json:
        new_msg = dict(msg)
        content = new_msg.get("content")

        if isinstance(content, str):
            if (image_token in content) or (video_token in content) or convert_plain_text:
                parts = [p for p in pattern.split(content) if p != ""]
                structured = []
                for p in parts:
                    if p == image_token:
                        structured.append({"type": "image"})
                    elif p == video_token:
                        structured.append({"type": "video"})
                    else:
                        structured.append({"type": "text", "text": p})
                new_msg["content"] = structured
        # If content is already a list or other type, leave it as-is
        out.append(new_msg)

    return out

def load_openai_tools_from_yaml(yaml_path: str) -> list[dict]:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # Each entry already matches OpenAI function tool schema: {"type":"function","function":{...}}
    return [t["tool_schema"] for t in data.get("tools", [])]

from typing import Iterable, Iterator, List, Optional, Union
from datasets import load_dataset

def iter_hf_parquet(
    source: Union[str, List[str]],
    *,
    split: str = "train",
    streaming: bool = True,
    columns: Optional[List[str]] = None,
) -> Iterator[dict]:
    """
    Iterate over rows from a Parquet-backed dataset.

    - source: one of
        - HF repo id (e.g., "owner/dataset_name")
        - local/remote parquet path ("/abs/path/file.parquet" or URL)
        - list of parquet paths
    - split: dataset split (when loading by repo id)
    - streaming: True to stream without full download (recommended for large data)
    - columns: optional subset of columns to keep
    """
    # If given parquet file(s), use the parquet builder; otherwise treat as repo id
    is_parquet = (
        (isinstance(source, str) and source.endswith(".parquet")) or
        (isinstance(source, list) and all(isinstance(s, str) and s.endswith(".parquet") for s in source))
    )

    if is_parquet:
        ds = load_dataset("parquet", data_files=source, split=split, streaming=streaming)
    else:
        ds = load_dataset(source, split=split, streaming=streaming)

    if columns is not None:
        ds = ds.select_columns(columns)

    for row in ds:
        yield row

import re

def parse_gpt_style_conversation(raw_text: str):
    """
    Convert GPT-style serialized conversation into a list of {role, content} dicts.
    Removes Qwen special tokens like <|im_start|> and <|im_end|>, 
    but keeps <image> tags unchanged.
    """
    # Step 1: Remove Qwen special tokens (im_start, im_end)
    cleaned = raw_text.replace("<|im_start|>", "|img_start|").replace("<|im_end|>", "|img_end|")

    # Step 2: Split by role markers
    pattern = re.compile(r"\|img_start\|\s*(\w+)\s*(.*?)\|img_end\|", re.DOTALL)
    messages = []
    for match in pattern.finditer(cleaned):
        role, content = match.groups()
        # strip unnecessary whitespace
        role = role.strip()
        content = content.strip()
        # restore proper <image> tags and remove artifacts
        content = re.sub(r"\|img_start\||\|img_end\|", "", content)
        messages.append({"role": role, "content": content})
    
    return messages

import os
from collections import defaultdict
from typing import Set, Dict, Any

def map_key_frame_indexes_to_interval_index(row: Dict[str, Any]) -> Set[int]:
    """
    From a row containing:
      - extra_info.key_frame_indexes: indices into extra_info.image_paths that are key frames
      - extra_info.image_paths: list of absolute/relative paths for all frames
      - non_key_frame_image_paths: dict[int, list[str]] mapping interval index -> list of frame paths within that interval

    Returns:
      A set of interval indices that contain the most key-frame occurrences (ties included).
      Returns an empty set if no key frame names are found within any interval lists.
    """
    key_frame_indexes = row['extra_info']['key_frame_indexes']
    image_paths = row['extra_info']['image_paths']
    image_names = [os.path.basename(path) for path in image_paths]
    key_frame_names = [image_names[i] for i in key_frame_indexes]

    frame_paths_by_interval = row['non_key_frame_image_paths']
    frame_names_by_interval = {
        k: [os.path.basename(path) for path in v]
        for k, v in frame_paths_by_interval.items()
    }

    key_interval_counts = defaultdict(int)
    for key_frame_name in key_frame_names:
        for interval, frame_names in frame_names_by_interval.items():
            if key_frame_name in frame_names:
                key_interval_counts[interval] += 1
                break  # key frame appears in at most one interval's list

    if not key_interval_counts:
        return set()

    max_count = max(key_interval_counts.values())
    if max_count == 0:
        return set()

    # Return all intervals tied for the maximum count
    return [interval for interval, cnt in key_interval_counts.items() if cnt == max_count][0]

def prepare_messages(row: dict) -> list[dict]:
    messages = [
        # {
        #     "role": "system",
        #     "content": (
        #         "You are a helpful assistant. You can call functions to assist with the user query. "
        #         "Important: You must call only one function at a time. After each function call, "
        #         "wait for the execution result before making the next function call if needed."
        #     ),
        # },
        {"role": "user", "content": row["prompt"][0]["content"].replace("image_resize_tool", "temporal_zoom_tool")},
    ]

    if 'key_frame_indexes' in row['extra_info']:
        key_interval_index = map_key_frame_indexes_to_interval_index(row)
        tool_call = {
            "name": "temporal_zoom_tool",
            "arguments": {"interval_index": key_interval_index}
        }

        try:
            message_tool_call = [
                {
                    "role": "assistant",
                    "content": (
                        "<think>I need to zoom in on the key frames to answer the user query.</think>"
                        f"<tool_call>{json.dumps(tool_call)}</tool_call>"
                    ),
                    "tool_calls": []
                }
            ]
        except Exception as e:
            return None

        messages.extend(message_tool_call)

        message_tool_response = [
            {
                "role": "user",
                "content": f"<tool_response><image><image><image><image><image><image><image><image>Zoomed in on the frames between Frame-{key_interval_index} and Frame-{int(key_interval_index) + 1}.</tool_response>"
            }
        ]
        messages.extend(message_tool_response)

    answer = json.loads(row['extra_info']['answer'])[0]
    message_answer = [
        {
            "role": "assistant",
            "content": f"<think>I reviewed the frames and can answer the user query now.</think><answer>{answer}</answer>",
            "tool_calls": []
        }
    ]
    messages.extend(message_answer)

    tools = load_openai_tools_from_yaml("temporal_zoom_tool_config.yaml")
    rendered_messages = render_chat_with_template(
        messages,
        tools=tools,
        add_generation_prompt=False,
        add_vision_id=True,
        normalize=False,
        template_str=CUSTOM_TEMPLATE,
    )

    rendered_messages = parse_gpt_style_conversation(rendered_messages)

    images = [item['image'] for item in row['images']]

    if 'key_frame_indexes' in row['extra_info']:
        images.extend(row['non_key_frame_image_paths'][key_interval_index])

    
    return {
        "messages": rendered_messages,
        "images": images
    }

if __name__ == "__main__":
    # dataset = iter_hf_parquet(
    #     "/home/geng/data/verl/scanqa_images_16_keyframes_120_non_keyframes_504x504/train_with_key_frames.parquet", 
    #     split="train")
    # results = []
    # for row in tqdm(dataset):
    #     results.append(prepare_messages(row))
    # with open("../scanqa_images_16_keyframes_120_non_keyframes_504x504_train_with_key_frames.json", "w") as f:
    #     for result in results:
    #        json.dump(result, f)

    dataset = iter_hf_parquet(
        "/home/geng/data/verl/sqa3d_images_16_keyframes_120_non_keyframes_504x504_with_label/train.parquet", 
        split="train")
    results = []
    for row in tqdm(dataset):
        tmp = prepare_messages(row)
        if tmp is not None:
            results.append(tmp)
        else:
            print('skipping row:', row)

    with open("../sqa3d_images_16_keyframes_120_non_keyframes_504x504_train.json", "w") as f:
        for result in results:
           json.dump(result, f)
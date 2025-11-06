import os
import re
import json
import yaml
import torch
import pprint
from collections import defaultdict
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm
from pprint import pprint
import json
from typing import Any, Iterator
from typing import List, Optional, Union
from datasets import load_dataset

CUSTOM_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{%- if tools %}{{- '<|im_start|>system\\n' }}{%- if messages[0]['role'] == 'system' %}{%- if messages[0]['content'] is string %}{{- messages[0]['content'] }}{%- else %}{{- messages[0]['content'][0]['text'] }}{%- endif %}{%- else %}{{- 'You are a helpful assistant.' }}{%- endif %}{{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}{%- for tool in tools %}{{- \"\\n\" }}{{- tool | tojson }}{%- endfor %}{{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}{% for message in messages %}{% if message['role'] != 'system' or loop.first == false %}{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{%- elif message.role == \"assistant\" %}{{- '<|im_start|>' + message.role }}{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- '\\n<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- '<|im_start|>user' }}{%- endif %}{{- '\\n<tool_response>\\n' }}{% if message['content'] is string %}{{ message.content }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'text' or 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{- '\\n</tool_response>' }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}{% endif %}{% endfor %}{%- else %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{%- elif message.role == \"assistant\" %}{{- '<|im_start|>' + message.role }}{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- '\\n<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}{%- endfor %}{{- '<|im_end|>\\n' }}{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- '<|im_start|>user' }}{%- endif %}{{- '\\n<tool_response>\\n' }}{% if message['content'] is string %}{{ message.content }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'text' or 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{- '\\n</tool_response>' }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endif %}{% endfor %}{%- endif %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

def load_jsonl(path: str, *, encoding: str = "utf-8", skip_empty: bool = True) -> list[Any]:
    with open(path, "r", encoding=encoding) as f:
        if skip_empty:
            return [json.loads(line) for line in f if line.strip()]
        return [json.loads(line) for line in f]

def iter_jsonl(path: str, *, encoding: str = "utf-8", skip_empty: bool = True) -> Iterator[Any]:
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            if skip_empty and not line.strip():
                continue
            yield json.loads(line)

from typing import Any, Dict, List, Tuple

def _index_by_qid_interval(rows: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    idx: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        k = (r["question_id"], r["interval"])
        idx.setdefault(k, []).append(r)
    return idx

def match_jsonl_on_qid_interval(path_a: str, path_b: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Returns list of (row_a, row_b) where both share the same (question_id, interval).
    Supports duplicates by returning all combinations for the same key.
    """
    a = load_jsonl(path_a)
    b = load_jsonl(path_b)
    idx_a = _index_by_qid_interval(a)

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for rb in b:
        k = (rb["question_id"], rb["interval"])
        if k in idx_a:
            for ra in idx_a[k]:
                pairs.append((ra, rb))
    return pairs

def join_jsonl_on_qid_interval(path_a: str, path_b: str) -> List[Dict[str, Any]]:
    """
    Inner-join: merges dicts for matched pairs. For key collisions with different values,
    keeps value from A and writes B’s value under '<key>_b'.
    """
    out: List[Dict[str, Any]] = []
    for ra, rb in match_jsonl_on_qid_interval(path_a, path_b):
        merged = dict(ra)
        for k, v in rb.items():
            if k in merged and merged[k] != v:
                merged[f"{k}_b"] = v
            else:
                merged[k] = v
        out.append(merged)
    return out

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
        {"role": "user", "content": row['parquet_data']["prompt"][0]["content"].replace("image_resize_tool", "temporal_zoom_tool")},
    ]

    key_interval_index = row['interval']
    tool_call = {
        "name": "temporal_zoom_tool",
        "arguments": {"interval_index": key_interval_index}
    }

    tool_call_thinking = row['thinking']
    message_tool_call = [
        {
            "role": "assistant",
            "content": (
                f"<think>{tool_call_thinking}</think>"
                f"<tool_call>{json.dumps(tool_call)}</tool_call>"
            ),
            "tool_calls": []
        }
    ]
    messages.extend(message_tool_call)

    message_tool_response = [
        {
            "role": "tool",
            "content": f"<tool_response><image><image><image><image><image><image><image><image>Zoomed in on the frames between Frame-{key_interval_index} and Frame-{int(key_interval_index) + 1}.</tool_response>"
        }
    ]
    messages.extend(message_tool_response)

    answer = json.loads(row['parquet_data']['extra_info']['answer'])[0]
    answer_thinking = row['summary']
    message_answer = [
        {
            "role": "assistant",
            "content": f"<think>{answer_thinking}</think><answer>{answer}</answer>",
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

    images = [item['image'] for item in row['parquet_data']['images']]

    if isinstance(key_interval_index, int):
        key_interval_index = str(key_interval_index)
    images.extend(row['parquet_data']['non_key_frame_image_paths'][key_interval_index])

    
    return {
        "messages": rendered_messages,
        "images": images
    }

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

def prepare_user_message(question, tool_name='temporal_zoom_tool'):
    template = f"""Frame-0: <image>\nFrame-1: <image>\nFrame-2: <image>\nFrame-3: <image>\nFrame-4: <image>\nFrame-5: <image>\nFrame-6: <image>\nFrame-7: <image>\nFrame-8: <image>\nFrame-9: <image>\nFrame-10: <image>\nFrame-11: <image>\nFrame-12: <image>\nFrame-13: <image>\nFrame-14: <image>\nFrame-15: <image>\nAnswer the question: {question}, Think first, call **{tool_name}** if needed, then answer. Format strictly as <think>...</think><tool_call>...</tool_call>(if tools needed)<answer>...</answer>.The answer within <answer>...</answer> should be a short phrase, e.g., 'brown sofa'."""

    return template

def load_openai_tools_from_yaml(yaml_path: str) -> list[dict]:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # Each entry already matches OpenAI function tool schema: {"type":"function","function":{...}}
    return [t["tool_schema"] for t in data.get("tools", [])]

def load_parquet_as_list(
    source: Union[str, List[str]],
    *,
    split: str = "train",
    columns: Optional[List[str]] = None,
) -> List[dict]:
    """
    Load rows from parquet and return as a list of dicts.
    - source: path, URL, or list of parquet files; or an HF repo id.
    - split: dataset split name (for parquet builder, 'train' is standard).
    - columns: optionally select a subset of columns.
    """
    is_parquet = (
        (isinstance(source, str) and source.endswith(".parquet")) or
        (isinstance(source, list) and all(isinstance(s, str) and s.endswith(".parquet") for s in source))
    )
    if is_parquet:
        ds = load_dataset("parquet", data_files=source, split=split, streaming=False)
    else:
        ds = load_dataset(source, split=split, streaming=False)

    if columns is not None:
        ds = ds.select_columns(columns)

    return list(ds)


if __name__ == "__main__":
    IF_UNIQUE = True

    PATH_PARQUET = "/home/geng/data/verl/scanqa_images_16_keyframes_120_non_keyframes_504x504_with_label/train.parquet"
    PATH_REASONING = "/home/geng/git/LLaMA-Factory/data/scanqa/reasoning.jsonl"
    PATH_ANSWER = "/home/geng/git/LLaMA-Factory/data/scanqa/answer.jsonl"
    PATH_OUTPUT = "../scanqa_images_16_keyframes_120_non_keyframes_504x504_train_with_key_frames_thinking_unique.json"

    parquet_data = load_parquet_as_list(
        PATH_PARQUET,
        split="train",
    )
    # pprint(parquet_data[0])

    pairs = match_jsonl_on_qid_interval(
        PATH_REASONING,
        PATH_ANSWER,
    )
    # pprint(pairs[0][0])
    # pprint(pairs[0][1])

    thinking_data = []
    for p in pairs:
        thinking_data.append(p[1])

    if IF_UNIQUE:
        # keep only the first record per question_id
        _unique = {}
        for t in thinking_data:
            qid = t.get('question_id')
            if qid not in _unique:
                _unique[qid] = t
        thinking_data = list(_unique.values())

    for t in thinking_data:
        t['parquet_data'] = parquet_data[t['row_index']]
    
    # pprint(thinking_data[-1])

    results = []
    for t in tqdm(thinking_data):
        results.append(prepare_messages(t))
    with open(PATH_OUTPUT, "w") as f:
        for result in results:
           json.dump(result, f)

    pprint(results[-1])

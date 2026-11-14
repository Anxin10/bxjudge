"""Pinned LocateAnything 3B image grounding for local prelabels."""

import hashlib
import json
import re
from pathlib import Path

MODEL_ID = "nvidia/LocateAnything-3B"
MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
MODEL_VERSION = f"locateanything-3b-{MODEL_REVISION}"
MODEL_MANIFEST = Path(__file__).resolve().parents[3] / "scripts/locateanything-runtime/model-sha256.json"
_COORDINATES = re.compile(r"<([0-9]{1,4})><([0-9]{1,4})><([0-9]{1,4})><([0-9]{1,4})>")


def parse_boxes(answer, prompts, width, height):
    """Decode tagged normalized XYXY boxes using exact prompt labels."""
    by_label = {phrase.strip().casefold(): cls for cls, phrase in prompts.items()}
    boxes = []
    for section in answer.split("<ref>")[1:]:
        phrase, separator, content = section.partition("</ref>")
        cls = by_label.get(phrase.strip().casefold())
        if not separator or cls is None:
            continue
        for item in re.findall(r"<box>([^\n]*?)</box>", content):
            match = _COORDINATES.fullmatch(item)
            if match is None:
                continue
            x0, y0, x1, y1 = map(int, match.groups())
            if 0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000:
                boxes.append((cls, (x0 * width / 1000, y0 * height / 1000,
                                    x1 * width / 1000, y1 * height / 1000), None))
    return boxes


def parse_response(answer, prompts, width, height):
    if not isinstance(answer, str) or not answer.rstrip().endswith("<|im_end|>"):
        raise ValueError("LocateAnything response did not finish; check max_new_tokens")
    return parse_boxes(answer, prompts, width, height)


def verify_model(directory: Path) -> None:
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if (manifest["repo"], manifest["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("LocateAnything manifest does not match the runtime revision")
    # Additional tokenizer or model files can change how Transformers loads
    # this directory. Only the Hub download metadata is outside the manifest.
    expected = manifest["files"]
    for file in directory.iterdir():
        if file.name not in expected and not (file.name == ".cache" and file.is_dir()):
            raise ValueError(f"Unexpected LocateAnything model file: {file}")
    for name, digest in expected.items():
        with (directory / name).open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"LocateAnything snapshot checksum mismatch: {directory / name}")


class LocateAnything:
    def __init__(self, config):
        directory = Path(config["model_dir"])
        verify_model(directory)

        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        options = {"local_files_only": True, "trust_remote_code": True}
        self.tokenizer = AutoTokenizer.from_pretrained(directory, **options)
        self.processor = AutoProcessor.from_pretrained(directory, **options)
        self.model = AutoModel.from_pretrained(
            directory, dtype=torch.bfloat16, use_safetensors=True, **options,
        ).to("cuda").eval()
        self.config = config
        self.question = (
            "Locate all the instances that matches the following description: "
            + "</c>".join(config["prompts"].values()) + "."
        )

    def __call__(self, frame):
        import cv2
        import torch
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": self.question},
        ]}]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
        torch.manual_seed(0)
        with torch.inference_mode():
            answer = self.model.generate(
                pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs["image_grid_hws"], tokenizer=self.tokenizer,
                max_new_tokens=self.config["max_new_tokens"], use_cache=True,
                generation_mode="hybrid", do_sample=True, temperature=0.7,
                top_p=0.9, repetition_penalty=1.1, verbose=False,
            )
        return parse_response(answer, self.config["prompts"], *image.size)

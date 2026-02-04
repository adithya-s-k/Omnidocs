"""MinerU VL utilities for document extraction.

Contains data structures, parsing, prompts, and post-processing functions
for MinerU VL document extraction pipeline.
"""

import html
import itertools
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============= Block Types =============


class BlockType(str, Enum):
    """MinerU VL block types (22 categories)."""

    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    ALGORITHM = "algorithm"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    PAGE_FOOTNOTE = "page_footnote"
    ASIDE_TEXT = "aside_text"
    EQUATION = "equation"
    EQUATION_BLOCK = "equation_block"
    REF_TEXT = "ref_text"
    LIST = "list"
    PHONETIC = "phonetic"
    TABLE_CAPTION = "table_caption"
    IMAGE_CAPTION = "image_caption"
    CODE_CAPTION = "code_caption"
    TABLE_FOOTNOTE = "table_footnote"
    IMAGE_FOOTNOTE = "image_footnote"
    UNKNOWN = "unknown"


# Set of valid block type values
BLOCK_TYPES = {bt.value for bt in BlockType}

# Valid rotation angles
ANGLE_OPTIONS = {None, 0, 90, 180, 270}


# ============= Content Block =============


class ContentBlock(BaseModel):
    """A detected content block with type, bounding box, angle, and content.

    Coordinates are normalized to [0, 1] range relative to image dimensions.
    """

    model_config = ConfigDict(extra="forbid")

    type: BlockType = Field(..., description="Block type")
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x1, y1, x2, y2] normalized to [0, 1]",
    )
    angle: Literal[None, 0, 90, 180, 270] = Field(
        default=None, description="Rotation angle of the block"
    )
    content: Optional[str] = Field(
        default=None,
        description="Extracted content (text, HTML table, LaTeX equation)",
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        if not all(0 <= coord <= 1 for coord in v):
            raise ValueError("Bbox coordinates must be in [0, 1] range")
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError("Invalid bbox: x1 < x2 and y1 < y2 required")
        return v

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def to_absolute(self, image_width: int, image_height: int) -> List[int]:
        """Convert normalized bbox to absolute pixel coordinates."""
        x1, y1, x2, y2 = self.bbox
        return [
            int(x1 * image_width),
            int(y1 * image_height),
            int(x2 * image_width),
            int(y2 * image_height),
        ]


# ============= Sampling Parameters =============


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    max_new_tokens: Optional[int] = None


class MinerUSamplingParams(SamplingParams):
    """Default sampling parameters optimized for MinerU VL."""

    def __init__(
        self,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = 0.01,
        top_k: Optional[int] = 1,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0,
        repetition_penalty: Optional[float] = 1.0,
        no_repeat_ngram_size: Optional[int] = 100,
        max_new_tokens: Optional[int] = None,
    ):
        super().__init__(
            temperature,
            top_p,
            top_k,
            presence_penalty,
            frequency_penalty,
            repetition_penalty,
            no_repeat_ngram_size,
            max_new_tokens,
        )


# ============= Prompts =============

SYSTEM_PROMPT = "You are a helpful assistant."
LAYOUT_PROMPT = "\nLayout Detection:"

RECOGNITION_PROMPTS = {
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
    "[default]": "\nText Recognition:",
}

DEFAULT_PROMPTS = {
    **RECOGNITION_PROMPTS,
    "[layout]": LAYOUT_PROMPT,
}

DEFAULT_SAMPLING_PARAMS = {
    "table": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.005),
    "equation": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[default]": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[layout]": MinerUSamplingParams(),
}


# ============= Parsing =============

LAYOUT_REGEX = (
    r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|>"
    r"<\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$"
)

ANGLE_MAPPING = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}


def convert_bbox(bbox: Sequence) -> Optional[List[float]]:
    """Convert bbox from model output (0-1000) to normalized format (0-1)."""
    bbox = tuple(map(int, bbox))
    if any(coord < 0 or coord > 1000 for coord in bbox):
        return None
    x1, y1, x2, y2 = bbox
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return [coord / 1000.0 for coord in (x1, y1, x2, y2)]


def parse_angle(tail: str) -> Literal[None, 0, 90, 180, 270]:
    """Parse rotation angle from model output tail string."""
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            return angle
    return None


def parse_layout_output(output: str) -> List[ContentBlock]:
    """Parse layout detection model output into ContentBlocks."""
    blocks = []
    for line in output.split("\n"):
        match = re.match(LAYOUT_REGEX, line)
        if not match:
            continue
        x1, y1, x2, y2, ref_type, tail = match.groups()
        bbox = convert_bbox((x1, y1, x2, y2))
        if bbox is None:
            continue
        ref_type = ref_type.lower()
        if ref_type not in BLOCK_TYPES:
            continue
        angle = parse_angle(tail)
        blocks.append(
            ContentBlock(
                type=BlockType(ref_type),
                bbox=bbox,
                angle=angle,
            )
        )
    return blocks


# ============= Image Utilities =============

LAYOUT_IMAGE_SIZE = (1036, 1036)


def get_rgb_image(image: Image.Image) -> Image.Image:
    """Convert image to RGB mode."""
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def prepare_for_layout(
    image: Image.Image,
    layout_size: Tuple[int, int] = LAYOUT_IMAGE_SIZE,
) -> Image.Image:
    """Prepare image for layout detection."""
    image = get_rgb_image(image)
    image = image.resize(layout_size, Image.Resampling.BICUBIC)
    return image


def resize_by_need(
    image: Image.Image,
    min_edge: int = 28,
    max_ratio: float = 50,
) -> Image.Image:
    """Resize image if needed based on aspect ratio constraints."""
    edge_ratio = max(image.size) / min(image.size)
    if edge_ratio > max_ratio:
        width, height = image.size
        if width > height:
            new_w, new_h = width, math.ceil(width / max_ratio)
        else:
            new_w, new_h = math.ceil(height / max_ratio), height
        new_image = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
        new_image.paste(image, (int((new_w - width) / 2), int((new_h - height) / 2)))
        image = new_image
    if min(image.size) < min_edge:
        scale = min_edge / min(image.size)
        new_w, new_h = round(image.width * scale), round(image.height * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return image


def prepare_for_extract(
    image: Image.Image,
    blocks: List[ContentBlock],
    prompts: Dict[str, str] = None,
    sampling_params: Dict[str, SamplingParams] = None,
    skip_types: set = None,
) -> Tuple[List[Image.Image], List[str], List[SamplingParams], List[int]]:
    """Prepare cropped images for content extraction."""
    if prompts is None:
        prompts = DEFAULT_PROMPTS
    if sampling_params is None:
        sampling_params = DEFAULT_SAMPLING_PARAMS
    if skip_types is None:
        skip_types = {"image", "list", "equation_block"}

    image = get_rgb_image(image)
    width, height = image.size

    block_images = []
    prompt_list = []
    params_list = []
    indices = []

    for idx, block in enumerate(blocks):
        if block.type.value in skip_types:
            continue

        x1, y1, x2, y2 = block.bbox
        scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
        block_image = image.crop(scaled_bbox)

        if block_image.width < 1 or block_image.height < 1:
            continue

        if block.angle in [90, 180, 270]:
            block_image = block_image.rotate(block.angle, expand=True)

        block_image = resize_by_need(block_image)
        block_images.append(block_image)

        block_type = block.type.value
        prompt = prompts.get(block_type) or prompts.get("[default]")
        prompt_list.append(prompt)

        params = sampling_params.get(block_type) or sampling_params.get("[default]")
        params_list.append(params)
        indices.append(idx)

    return block_images, prompt_list, params_list, indices


# ============= OTSL to HTML Conversion =============

OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"
ALL_OTSL_TOKENS = [OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]


def convert_otsl_to_html(otsl_content: str) -> str:
    """Convert OTSL table format to HTML."""
    if otsl_content.startswith("<table") and otsl_content.endswith("</table>"):
        return otsl_content

    pattern = r"(" + r"|".join(ALL_OTSL_TOKENS) + r")"
    tokens = re.findall(pattern, otsl_content)
    text_parts = re.split(pattern, otsl_content)
    text_parts = [part for part in text_parts if part.strip()]

    split_row_tokens = [
        list(y) for x, y in itertools.groupby(tokens, lambda z: z == OTSL_NL) if not x
    ]
    if not split_row_tokens:
        return ""

    max_cols = max(len(row) for row in split_row_tokens)
    for row in split_row_tokens:
        while len(row) < max_cols:
            row.append(OTSL_ECEL)

    def count_right(tokens_grid, c, r, which_tokens):
        span = 0
        c_iter = c
        while c_iter < len(tokens_grid[r]) and tokens_grid[r][c_iter] in which_tokens:
            c_iter += 1
            span += 1
        return span

    def count_down(tokens_grid, c, r, which_tokens):
        span = 0
        r_iter = r
        while r_iter < len(tokens_grid) and tokens_grid[r_iter][c] in which_tokens:
            r_iter += 1
            span += 1
        return span

    table_cells = []
    r_idx = 0
    c_idx = 0

    for i, text in enumerate(text_parts):
        if text in [OTSL_FCEL, OTSL_ECEL]:
            row_span = 1
            col_span = 1
            cell_text = ""
            right_offset = 1

            if text != OTSL_ECEL and i + 1 < len(text_parts):
                next_text = text_parts[i + 1]
                if next_text not in ALL_OTSL_TOKENS:
                    cell_text = next_text
                    right_offset = 2

            if i + right_offset < len(text_parts):
                next_right = text_parts[i + right_offset]
                if next_right in [OTSL_LCEL, OTSL_XCEL]:
                    col_span += count_right(
                        split_row_tokens, c_idx + 1, r_idx, [OTSL_LCEL, OTSL_XCEL]
                    )

            if r_idx + 1 < len(split_row_tokens) and c_idx < len(
                split_row_tokens[r_idx + 1]
            ):
                next_bottom = split_row_tokens[r_idx + 1][c_idx]
                if next_bottom in [OTSL_UCEL, OTSL_XCEL]:
                    row_span += count_down(
                        split_row_tokens, c_idx, r_idx + 1, [OTSL_UCEL, OTSL_XCEL]
                    )

            table_cells.append(
                {
                    "text": cell_text.strip(),
                    "row_span": row_span,
                    "col_span": col_span,
                    "start_row": r_idx,
                    "start_col": c_idx,
                }
            )

        if text in [OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]:
            c_idx += 1
        if text == OTSL_NL:
            r_idx += 1
            c_idx = 0

    num_rows = len(split_row_tokens)
    num_cols = max_cols
    grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    for cell in table_cells:
        for i in range(
            cell["start_row"], min(cell["start_row"] + cell["row_span"], num_rows)
        ):
            for j in range(
                cell["start_col"], min(cell["start_col"] + cell["col_span"], num_cols)
            ):
                grid[i][j] = cell

    html_parts = []
    for i in range(num_rows):
        html_parts.append("<tr>")
        for j in range(num_cols):
            cell = grid[i][j]
            if cell is None:
                continue
            if cell["start_row"] != i or cell["start_col"] != j:
                continue

            content = html.escape(cell["text"])
            tag = "td"
            parts = [f"<{tag}"]
            if cell["row_span"] > 1:
                parts.append(f' rowspan="{cell["row_span"]}"')
            if cell["col_span"] > 1:
                parts.append(f' colspan="{cell["col_span"]}"')
            parts.append(f">{content}</{tag}>")
            html_parts.append("".join(parts))
        html_parts.append("</tr>")

    return f"<table>{''.join(html_parts)}</table>"


def simple_post_process(blocks: List[ContentBlock]) -> List[ContentBlock]:
    """Simple post-processing: convert OTSL tables to HTML."""
    for block in blocks:
        if block.type == BlockType.TABLE and block.content:
            try:
                block.content = convert_otsl_to_html(block.content)
            except Exception:
                pass
    return blocks

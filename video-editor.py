"""
TikTok Video AI Agent
- Transcribes video with word-level timestamps (faster-whisper)
- Generates TikTok-style ASS subtitles
- Removes silences/filler words using a local HuggingFace instruct model
- Burns subtitles and applies cuts via FFmpeg
"""

import os
import torch
import json
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline

load_dotenv()

if not torch.cuda.is_available():
    raise EnvironmentError(
        "No CUDA GPU detected. This config requires an NVIDIA GPU with 8GB+ VRAM.\n"
        "If running on CPU/MPS, set USE_4BIT=False and switch to a smaller model."
    )

@dataclass
class WordSegment:
    word: str
    start: float
    end: float


def transcribe(video_path: str):
    print(f"[1/4] Transcribing {video_path} on GPU ...")
    model = WhisperModel(os.getenv("WHISPER_MODEL"), device=os.getenv("DEVICE"), compute_type="float16")
    segments, _ = model.transcribe(video_path, word_timestamps=True)
    _words: list[WordSegment] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                _words.append(WordSegment(
                    word=w.word.strip(),
                    start=w.start,
                    end=w.end,
                ))

    print(f"    → {len(_words)} words transcribed.")
    return _words

# Module-level cache so the model is only loaded once per session
_llm_pipeline = None

def load_llm():
    global _llm_pipeline
    if _llm_pipeline is not None:
        return _llm_pipeline

    # 4-bit NF4 config: best quality/size trade-off for inference
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4 GB extra
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if os.getenv("USE_4BIT") else None

    llm_model = os.getenv("LLM_MODEL")
    tokenizer = AutoTokenizer.from_pretrained(llm_model)

    model = AutoModelForCausalLM.from_pretrained(
        llm_model,
        quantization_config=quant_config,  # 4-bit NF4 when USE_4BIT=True
        device_map="auto",  # auto-assigns layers across GPU/CPU
        dtype=torch.bfloat16,  # compute dtype for non-quantized layers
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",  # faster attention on Ampere+ (30xx/40xx)
    )
    model.eval()

    _llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,  # greedy decoding — deterministic JSON output
        temperature=None,
        top_p=None,
        return_full_text=False,  # return only the newly generated tokens, not the prompt
    )

    # Log actual VRAM usage after loading
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"    ✓ Model loaded — VRAM: {allocated:.1f}GB allocated / {reserved:.1f}GB reserved")

    return _llm_pipeline

def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged

def agent_decide_cuts(words: list[WordSegment]) -> list[tuple[float, float]]:
    print("[2/4] Running AI agent to plan cuts")

    transcript_json = json.dumps(
        [{"w": ws.word, "s": round(ws.start, 2), "e": round(ws.end, 2)} for ws in words],
        ensure_ascii=False,
    )

    filler_words = {
        "é", "eh", "ahn", "hã", "hum", "hmm", "uh", "um",
        "tipo", "assim", "né", "né?", "tá", "tá?",
        "então", "bom", "cara", "mano", "meio que",
        "vamos dizer", "digamos", "basicamente",
        "literalmente", "na verdade", "quer dizer",
        "sabe", "sabe?", "sei lá"
    }

    system_prompt = (
        "You are a video editor assistant. "
        "When given a word-level transcript, you respond ONLY with a valid JSON array. "
        "No explanation, no markdown, no extra text — just the JSON array."
    )
    user_prompt = f"""
    Given this word-level transcript of a TikTok video, identify segments to CUT to make the video snappier.

    Rules:
    1. Remove filler words: {",".join(filler_words)} (only when used as filler).
    2. Remove pauses longer than 0.6 seconds between consecutive words.
    3. Do NOT cut meaningful content.
    4. Return ONLY a JSON array of objects with "start" and "end" float keys (seconds).
       Example: [{{"start": 1.2, "end": 1.8}}, {{"start": 4.0, "end": 4.7}}]
       If nothing to cut, return: []

    Transcript:
    {transcript_json}"""

    pipe = load_llm()
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    output = pipe(message)
    raw = output[0]["generated_text"]

    if isinstance(raw, list):
        raw = raw[-1].get("content","") if isinstance(raw[-1],dict) else str(raw[-1])

   # Extract JSON array from the response
    cuts: list[tuple[float, float]] = []
    try:
        start_idx = raw.find("[")
        end_idx   = raw.rfind("]") + 1
        if start_idx != -1 and end_idx > start_idx:
            parsed = json.loads(raw[start_idx:end_idx])
            cuts = [(c["start"], c["end"]) for c in parsed if "start" in c and "end" in c]
    except Exception as e:
        print(f"    ⚠ Could not parse LLM cuts: {e}. Will rely on pause detection only.")

     # Augment with automatic long-pause detection (always runs as a safety net)
    for i in range(len(words) - 1):
        gap = words[i + 1].start - words[i].end
        if gap > 0.7:
            cuts.append((words[i].end + 0.05, words[i + 1].start - 0.05))

    cuts = merge_intervals(cuts)
    print(f"    → {len(cuts)} cut(s) planned.")
    return cuts

def seconds_to_ass(t: float) -> str:
    """Convert seconds to ASS timestamp format H:MM:SS.cc"""
    h  = int(t // 3600)
    m  = int((t % 3600) // 60)
    s  = int(t % 60)
    cs = int(round((t % 1) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def build_ass(words: list[WordSegment], cuts: list[tuple[float, float]], output_path: str):
    """
    Words are grouped in chunks of CHUNK_SIZE.
    Cut segments are excluded.
    """
    print("[3/4] Generating TikTok-style subtitles ...")

    chunk_size = 6
    font_name     = "Arial"
    font_size     = 80            # in ASS points (scales to ~72px on 1080p)
    primary_color = "&H00FFFFFF"  # white
    outline_color = "&H00000000"  # black
    back_color    = "&H00000000"
    bold          = 1
    outline       = 4             # stroke thickness
    shadow        = 0

    def is_cut(t: float) -> bool:
        return any(s <= t <= e for s, e in cuts)

    # Filter out words that fall in cut zones
    visible_words = [w for w in words if not is_cut((w.start + w.end) / 2)]

    # Group into chunks
    chunks = [visible_words[i:i+chunk_size] for i in range(0, len(visible_words), chunk_size)]

    header = f"""\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTok,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},{back_color},{bold},0,0,0,100,100,0,0,1,{outline},{shadow},2,10,10,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = []
    for chunk in chunks:
        if not chunk:
            continue
        t_start = chunk[0].start
        t_end   = chunk[-1].end
        # Build karaoke-style highlighted text using {\k} tags
        text_parts = []
        for w in chunk:
            duration_cs = int(round((w.end - w.start) * 100))
            text_parts.append(f"{{\\k{duration_cs}}}{w.word} ")
        text = "".join(text_parts).strip()

        # Wrap in position tag — center horizontally, lower third
        text = r"{\an2\pos(540,1700)\b1}" + text

        line = (
            f"Dialogue: 0,{seconds_to_ass(t_start)},{seconds_to_ass(t_end)},"
            f"TikTok,,0,0,0,,{text}"
        )
        lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines))

    print(f"    → Subtitles written to {output_path}")

def apply_cuts_and_burn(
    input_video: str,
    ass_path: str,
    cuts: list[tuple[float, float]],
    output_path: str,
):
    """
    Uses FFmpeg to:
    1. Remove cut segments via select/aselect filters
    2. Burn ASS subtitles into the video
    """
    print("[4/4] Applying cuts and burning subtitles with FFmpeg ...")

    print(f"Applying {len(cuts)} cuts...")

    if cuts:
        # Build FFmpeg select filter to keep non-cut segments
        select_parts = []
        for start, end in cuts:
            select_parts.append(f"between(t,{start},{end})")
        cut_expr = "+".join(select_parts)
        keep_expr = f"not({cut_expr})"

        vf = (
            f"select='{keep_expr}',setpts=N/FRAME_RATE/TB,"
            f"ass={ass_path}"
        )
        af = f"aselect='{keep_expr}',asetpts=N/SR/TB"

        cmd = [
            "ffmpeg", "-y", "-i", input_video,
            "-vf", vf,
            "-af", af,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
    else:
        # No cuts — just burn subtitles
        cmd = [
            "ffmpeg", "-y", "-i", input_video,
            "-vf", f"ass={ass_path}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg error:\n", result.stderr)
        raise RuntimeError("FFmpeg failed.")
    print(f"    → Output saved to {output_path}")

def process_video(input_path: str, output_path: str = None):
    input_path = str(Path(input_path).resolve())
    if not output_path:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}_edited.mp4")

    with tempfile.TemporaryDirectory() as tmp:
        ass_path = os.path.join(tmp, "subtitles.ass")

        words = transcribe(input_path)
        cuts = agent_decide_cuts(words)
        build_ass(words, cuts, ass_path)
        apply_cuts_and_burn(input_path, ass_path, cuts, output_path)

    print(f"\n✅ Done! Edited video: {output_path}")
    return output_path

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video-editor.py <input_video.mp4> [output.mp4]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    process_video(inp, out)
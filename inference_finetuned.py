import argparse
import os
import random

import numpy as np
import soundfile as sf
import torch

from dia.model import Dia


# Mapping speaker names to byte values (must match training LANG2BYTE)
SPEAKER_MAP = {
    "Hương Lý": 0,
    "Phương Anh": 1,
    "Trần Quyên": 2,
    "Châu Anh": 3,
    "Danh Sơn": 4,
}


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_text(text: str, speaker: str | None = None) -> str:
    """
    Prepares text input by adding speaker prefix if provided.
    
    Args:
        text: Input text to synthesize
        speaker: Speaker name (e.g., "Hương Lý")
        
    Returns:
        Formatted text with speaker prefix if applicable
    """
    if speaker:
        if speaker not in SPEAKER_MAP:
            available_speakers = ", ".join(SPEAKER_MAP.keys())
            raise ValueError(
                f"Unknown speaker: '{speaker}'. Available speakers: {available_speakers}"
            )
        return f"[{speaker}]{text}"
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio using fine-tuned Dia model."
    )

    # Required arguments
    parser.add_argument(
        "text",
        type=str,
        help="Input text for speech generation. Can include speaker prefix like '[Hương Lý]text'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the generated audio file (e.g., output.wav).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.json file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint (.pth or .safetensors).",
    )

    # Speaker argument
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        choices=list(SPEAKER_MAP.keys()),
        help=f"Speaker name. Available: {', '.join(SPEAKER_MAP.keys())}. "
        "If provided, will be added as prefix to text.",
    )

    # Audio prompt for voice cloning
    parser.add_argument(
        "--audio-prompt",
        type=str,
        default=None,
        help="Path to an optional audio prompt WAV file for voice cloning.",
    )

    # Generation parameters
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of audio tokens to generate (defaults to config value).",
    )
    gen_group.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="Classifier-Free Guidance scale (default: 3.0). Higher values = more adherence to text.",
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=1.3,
        help="Sampling temperature (default: 1.3). Higher = more random, lower = more deterministic.",
    )
    gen_group.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability (default: 0.95).",
    )
    gen_group.add_argument(
        "--cfg-filter-top-k",
        type=int,
        default=35,
        help="Top-K filtering for CFG (default: 35).",
    )

    # Infrastructure
    infra_group = parser.add_argument_group("Infrastructure")
    infra_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    infra_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g., 'cuda', 'cpu', default: auto).",
    )
    infra_group.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster inference (experimental).",
    )

    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.config):
        parser.error(f"Config file not found: {args.config}")
    if not os.path.exists(args.checkpoint):
        parser.error(f"Checkpoint file not found: {args.checkpoint}")
    if args.audio_prompt and not os.path.exists(args.audio_prompt):
        parser.error(f"Audio prompt file not found: {args.audio_prompt}")

    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Determine device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Prepare text with speaker prefix if needed
    try:
        prepared_text = prepare_text(args.text, args.speaker)
    except ValueError as e:
        parser.error(str(e))

    print(f"Input text: {prepared_text}")

    # Load model from fine-tuned checkpoint
    print(f"Loading model from checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    try:
        model = Dia.from_local(args.config, args.checkpoint, device=device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Generate audio
    print("Generating audio...")
    try:
        sample_rate = 44100

        output_audio = model.generate(
            text=prepared_text,
            audio_prompt_path=args.audio_prompt,
            max_tokens=args.max_tokens,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_p=args.top_p,
            use_torch_compile=args.compile,
            cfg_filter_top_k=args.cfg_filter_top_k,
        )
        print("Audio generation complete.")

        # Save audio
        print(f"Saving audio to {args.output}...")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        sf.write(args.output, output_audio, sample_rate)
        print(f"✓ Audio successfully saved to {args.output}")

    except Exception as e:
        print(f"Error during audio generation or saving: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

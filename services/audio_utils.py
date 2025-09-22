import io
import os
import uuid
import numpy as np
import ffmpeg
from typing import Union, Tuple, Optional
import struct
import logging

logger = logging.getLogger(__name__)


def detect_audio_format(audio_bytes: bytes) -> Optional[str]:
    if len(audio_bytes) < 12:
        return None

    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return "wav"
    elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return "mp3"
    elif audio_bytes[:4] == b"fLaC":
        return "flac"
    elif audio_bytes[:4] == b"OggS":
        return "ogg"
    elif audio_bytes[4:8] == b"ftyp":
        if b"M4A " in audio_bytes[8:20] or b"mp4a" in audio_bytes[:32]:
            return "m4a"
        elif b"isom" in audio_bytes[:32] or b"mp42" in audio_bytes[:32]:
            return "mp4"
    elif audio_bytes[:8] == b"\x1a\x45\xdf\xa3":
        return "webm"

    return None


def inspect_audio_bytes(audio_bytes: bytes) -> dict:
    info = {
        "length": len(audio_bytes),
        "first_16_bytes": (
            audio_bytes[:16].hex() if len(audio_bytes) >= 16 else audio_bytes.hex()
        ),
        "detected_format": detect_audio_format(audio_bytes),
        "is_likely_raw_pcm": False,
        "has_wav_header": False,
        "is_empty_or_zeros": False,
    }
    if len(audio_bytes) == 0:
        info["is_empty_or_zeros"] = True
    elif len(audio_bytes) < 100:
        info["is_empty_or_zeros"] = True
    else:
        zero_count = audio_bytes.count(0)
        zero_percentage = zero_count / len(audio_bytes)
        info["zero_percentage"] = zero_percentage
        if zero_percentage > 0.95:
            info["is_empty_or_zeros"] = True

    if len(audio_bytes) >= 12:
        info["has_wav_header"] = (
            audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"
        )

    if (
        info["detected_format"] is None
        and len(audio_bytes) > 1000
        and not info["is_empty_or_zeros"]
    ):
        info["is_likely_raw_pcm"] = True

    return info


def smart_audio_to_pcm(
    audio_bytes: bytes, target_sample_rate: int = 16000
) -> np.ndarray:
    inspection = inspect_audio_bytes(audio_bytes)
    try:
        return convert_any_audio_to_pcm(audio_bytes, "wav", target_sample_rate)
    except Exception as e:
        raise ValueError(
            f"Unable to convert audio data. Inspection: {inspection}. Last error: {str(e)}"
        )


def convert_any_audio_to_pcm(
    audio_bytes: bytes, audio_format: str = "wav", target_sample_rate: int = 16000
) -> np.ndarray:
    debug_dir = "debug_audio"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    file_extension = audio_format.lower()
    debug_filename = os.path.join(
        debug_dir, f"received_audio_{uuid.uuid4()}.{file_extension}"
    )
    with open(debug_filename, "wb") as f:
        f.write(audio_bytes)

    try:
        if audio_format.lower() == "wav":
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        elif audio_format.lower() == "mp3":
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        elif audio_format.lower() == "flac":
            audio_segment = AudioSegment.from_flac(io.BytesIO(audio_bytes))
        elif audio_format.lower() in ["m4a", "mp4"]:
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_bytes), format="mp4"
            )
        else:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))

        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)

        if audio_segment.frame_rate != target_sample_rate:
            audio_segment = audio_segment.set_frame_rate(target_sample_rate)

        raw_audio = audio_segment.raw_data
        audio_array = np.frombuffer(raw_audio, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0
        return audio_array

    except Exception as e:
        raise ValueError(f"Failed to convert {audio_format} bytes to PCM: {str(e)}")


def reconstruct_wav_from_chunks(chunks: list[bytes]) -> bytes:
    if not chunks:
        return b""

    first_chunk = chunks[0]
    if len(first_chunk) < 44:
        raise ValueError("First chunk is too small to contain a WAV header.")

    header = first_chunk[:44]
    data_size = sum(len(chunk) for chunk in chunks) - 44
    new_header = bytearray(header)
    riff_chunk_size = data_size + 36
    new_header[4:8] = struct.pack("<I", riff_chunk_size)
    new_header[40:44] = struct.pack("<I", data_size)
    full_audio = bytearray()
    full_audio.extend(new_header)
    if len(first_chunk) > 44:
        full_audio.extend(first_chunk[44:])

    for chunk in chunks[1:]:
        full_audio.extend(chunk)

    return bytes(full_audio)


def universal_audio_to_pcm(
    audio_bytes: bytes, target_sample_rate: int = 16000, target_channels: int = 1
) -> np.ndarray:
    """
    Universal audio conversion method that works with ANY audio format.
    Uses multiple fallback strategies to ensure conversion always succeeds.

    Strategy:
    1. Try FFmpeg with auto-detection (most robust)
    2. Try FFmpeg with detected format hint
    3. Try FFmpeg with common format assumptions
    4. Fallback to pydub (original method)
    5. Last resort: treat as raw PCM

    Args:
        audio_bytes: Raw audio data in any format
        target_sample_rate: Desired sample rate (default: 16000 Hz)
        target_channels: Desired number of channels (default: 1 for mono)

    Returns:
        np.ndarray: Audio as float32 array normalized to [-1, 1]

    Raises:
        ValueError: If all conversion methods fail
    """
    if not audio_bytes or len(audio_bytes) == 0:
        raise ValueError("Empty audio data provided")

    inspection = inspect_audio_bytes(audio_bytes)
    logger.info(
        f"Converting audio: {len(audio_bytes)} bytes, detected format: {inspection.get('detected_format', 'unknown')}"
    )
    debug_filename = None
    if logger.isEnabledFor(logging.DEBUG):
        debug_dir = "debug_audio"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        debug_filename = os.path.join(debug_dir, f"audio_{uuid.uuid4()}.bin")
        with open(debug_filename, "wb") as f:
            f.write(audio_bytes)
        logger.debug(f"Debug audio saved: {debug_filename}")

    errors = []

    # Strategy 1: FFmpeg with auto-detection
    try:
        return _ffmpeg_convert_auto(audio_bytes, target_sample_rate, target_channels)
    except Exception as e:
        errors.append(f"FFmpeg auto-detection: {str(e)}")
        logger.debug(f"FFmpeg auto-detection failed: {e}")

    # Strategy 2: FFmpeg with format hint
    detected_format = inspection.get("detected_format")
    if detected_format:
        try:
            return _ffmpeg_convert_with_format(
                audio_bytes, detected_format, target_sample_rate, target_channels
            )
        except Exception as e:
            errors.append(f"FFmpeg with {detected_format}: {str(e)}")
            logger.debug(f"FFmpeg with format {detected_format} failed: {e}")

    # Strategy 3: FFmpeg with common format assumptions
    common_formats = ["wav", "mp3", "flac", "m4a", "ogg", "webm"]
    for fmt in common_formats:
        if fmt != detected_format:  # Skip if we already tried this format
            try:
                return _ffmpeg_convert_with_format(
                    audio_bytes, fmt, target_sample_rate, target_channels
                )
            except Exception as e:
                errors.append(f"FFmpeg {fmt}: {str(e)}")
                logger.debug(f"FFmpeg {fmt} assumption failed: {e}")

    # Strategy 4: Last resort - treat as raw PCM
    try:
        return _raw_pcm_convert(audio_bytes, target_sample_rate, target_channels)
    except Exception as e:
        errors.append(f"Raw PCM: {str(e)}")
        logger.debug(f"Raw PCM fallback failed: {e}")

    error_summary = "; ".join(errors)
    raise ValueError(
        f"All audio conversion methods failed. "
        f"Audio info: {inspection}. "
        f"Errors: {error_summary}"
        + (f". Debug file: {debug_filename}" if debug_filename else "")
    )


def _ffmpeg_convert_auto(
    audio_bytes: bytes, target_sample_rate: int, target_channels: int
) -> np.ndarray:
    """FFmpeg conversion with auto format detection."""
    process = (
        ffmpeg.input("pipe:0")
        .output(
            "pipe:1",
            acodec="pcm_s16le",
            ac=target_channels,
            ar=target_sample_rate,
            f="s16le",
        )
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)
    )
    stdout, stderr = process.communicate(input=audio_bytes)
    if process.returncode != 0:
        stderr_msg = (
            stderr.decode("utf-8", errors="ignore") if stderr else "Unknown error"
        )
        raise ValueError(f"FFmpeg auto-detection failed: {stderr_msg}")

    if len(stdout) == 0:
        raise ValueError("FFmpeg produced no output")

    audio_array = np.frombuffer(stdout, dtype=np.int16)
    audio_array = audio_array.astype(np.float32) / 32768.0
    logger.info(f"FFmpeg auto-detection success: {len(audio_array)} samples")
    return audio_array


def _ffmpeg_convert_with_format(
    audio_bytes: bytes, input_format: str, target_sample_rate: int, target_channels: int
) -> np.ndarray:
    """FFmpeg conversion with explicit format specification."""
    process = (
        ffmpeg.input("pipe:0", format=input_format)
        .output(
            "pipe:1",
            acodec="pcm_s16le",
            ac=target_channels,
            ar=target_sample_rate,
            f="s16le",
        )
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)
    )
    stdout, stderr = process.communicate(input=audio_bytes)
    if process.returncode != 0:
        stderr_msg = (
            stderr.decode("utf-8", errors="ignore") if stderr else "Unknown error"
        )
        raise ValueError(f"FFmpeg {input_format} failed: {stderr_msg}")

    if len(stdout) == 0:
        raise ValueError(f"FFmpeg {input_format} produced no output")

    audio_array = np.frombuffer(stdout, dtype=np.int16)
    audio_array = audio_array.astype(np.float32) / 32768.0
    logger.info(f"FFmpeg {input_format} success: {len(audio_array)} samples")
    return audio_array


def _pydub_convert(audio_bytes: bytes, target_sample_rate: int) -> np.ndarray:
    """Pydub fallback conversion (existing method)."""
    return smart_audio_to_pcm(audio_bytes, target_sample_rate)


def _raw_pcm_convert(
    audio_bytes: bytes, target_sample_rate: int, target_channels: int
) -> np.ndarray:
    if len(audio_bytes) < 1000:
        raise ValueError("Audio data too short for raw PCM interpretation")

    data_start = 44 if len(audio_bytes) > 44 else 0
    pcm_data = audio_bytes[data_start:]
    for dtype, divisor in [
        (np.int16, 32768.0),
        (np.int32, 2147483648.0),
        (np.float32, 1.0),
    ]:
        try:
            for byteorder in ["<", ">"]:
                try:
                    if dtype == np.float32:
                        audio_array = np.frombuffer(pcm_data, dtype=f"{byteorder}f4")
                    else:
                        audio_array = np.frombuffer(
                            pcm_data, dtype=f"{byteorder}i{dtype().itemsize}"
                        )

                    if len(audio_array) == 0:
                        continue

                    if dtype != np.float32:
                        audio_array = audio_array.astype(np.float32) / divisor

                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0 and max_val < 100:
                        logger.info(
                            f"Raw PCM interpretation success: {len(audio_array)} samples, dtype={dtype}, byteorder={byteorder}"
                        )
                        return audio_array

                except (ValueError, OverflowError):
                    continue
        except Exception:
            continue

    raise ValueError("Could not interpret audio as raw PCM data")

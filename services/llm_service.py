import logging
from typing import Optional
from llama_cpp import Llama
from config import config

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.llm_model: Optional[Llama] = None
        self._load_model()

    def _load_model(self):
        try:
            llm_model_path = config.get_llm_model_path()
            logger.info(f"Loading LLM model from: {llm_model_path}")
            self.llm_model = Llama(
                model_path=llm_model_path,
                n_ctx=config.LLM_N_CTX,
                n_threads=config.LLM_N_THREADS,
                verbose=False,
            )
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {str(e)}")
            logger.warning("Text correction will fall back to basic formatting")
            self.llm_model = None

    def _create_correction_prompt(self, text: str) -> str:
        return f"""You are an expert text correction assistant specialized in fixing speech-to-text transcription errors. Your task is to correct grammar, spelling, punctuation, and formatting while preserving the original meaning and intent.

            Primary Corrections:
            • Fix spelling mistakes and speech recognition errors (their/there, to/too, etc.)
            • Add proper punctuation (periods, commas, question marks, exclamation points)
            • Correct capitalization (sentence beginnings, proper nouns)
            • Break up run-on sentences into readable segments
            • Remove excessive filler words only when they impair readability (um, uh, like, you know)
            • Fix grammar while maintaining natural speech patterns
            • Correct word boundaries and missing spaces

            Smart Formatting Commands:
            When you detect dictation commands with 80%+ confidence based on context, convert them appropriately:
            • "period" / "full stop" → .
            • "comma" → ,
            • "question mark" → ?
            • "exclamation point" / "exclamation mark" → !
            • "new line" / "new paragraph" → line break
            • "bullet point" / "dash" → • (when creating lists)
            • "number one, number two" → 1. 2. (when creating numbered lists)
            • "bold [text]" / "italic [text]" → apply only if clearly intentional formatting

            Quality Guidelines:
            • Preserve the speaker's voice, style, and personality
            • Keep technical terms, jargon, and proper nouns intact
            • Don't over-correct casual or conversational language
            • Maintain regional dialects and speech patterns where appropriate
            • Do not translate text in other languages; preserve non-English segments exactly as spoken.
            • If multiple interpretations are possible, choose the most contextually appropriate
            • If the text looks like an email then format it as an email

            Critical Rules:
            • Return ONLY the corrected text—no explanations, comments, markdown, suffix, prefix, or any other text.
            • Output plain text format only
            • Preserve the core message and meaning exactly
            • Retain the languages present in the input
            • When uncertain about a correction, err on the side of minimal changes
            • Never translate or alter words in any language other than English; retain all original foreign-language text verbatim.

            Please correct this transcribed text: {text}
            """

    def correct_text(self, text: str) -> str:
        if not self.llm_model or not text.strip():
            return text

        try:
            prompt = self._create_correction_prompt(text)
            result = self.llm_model(
                prompt,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                top_p=config.LLM_TOP_P,
                stop=["<|eot_id|>", "\n\n"],
                echo=False,
            )
            corrected_text = result["choices"][0]["text"].strip()
            if (
                not corrected_text
                or len(corrected_text) < len(text) * config.LLM_CORRECTION_THRESHOLD
            ):
                logger.warning(
                    "LLM correction resulted in unusually short text, using original"
                )
                return text
            return corrected_text

        except Exception as e:
            logger.error(f"LLM text correction failed: {str(e)}")
            return text

    def is_available(self) -> bool:
        return self.llm_model is not None

import logging
import queue
from typing import Optional, List
from llama_cpp import Llama
from config import config
from groq import Groq

logger = logging.getLogger(__name__)


class LLMModelInstance:
    def __init__(self, instance_id: int):
        self.instance_id = instance_id
        self.model: Optional[Llama] = None
        self._load_model()

    def _load_model(self):
        try:
            llm_model_path = config.get_llm_model_path()
            logger.info(
                f"Loading LLM model instance {self.instance_id} from: {llm_model_path}"
            )
            self.model = Llama(
                model_path=llm_model_path,
                n_ctx=config.LLM_N_CTX,
                n_threads=config.LLM_N_THREADS,
                n_batch=config.LLM_N_BATCH,
                n_ubatch=config.LLM_N_UBATCH,
                verbose=False,
                use_mlock=True,
                top_k=config.LLM_TOP_K,
                top_p=config.LLM_TOP_P,
                temperature=config.LLM_TEMPERATURE,
            )
            logger.info(f"LLM model instance {self.instance_id} loaded successfully")
        except Exception as e:
            logger.error(
                f"Failed to load LLM model instance {self.instance_id}: {str(e)}"
            )
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def correct(self, prompt: str) -> Optional[str]:
        if not self.model:
            return None

        try:
            result = self.model(
                prompt,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                top_p=config.LLM_TOP_P,
                stop=["<|eot_id|>", "\n\n"],
                echo=False,
            )
            return result["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Instance {self.instance_id} generation failed: {str(e)}")
            return None


class LLMService:
    def __init__(self):
        self.pool_size = config.LLM_POOL_SIZE
        self.model_instances: List[LLMModelInstance] = []
        self.instance_queue: queue.Queue = queue.Queue()
        self._load_model_pool()
        self.warmup()
        self.groq: Groq = Groq(api_key=config.GROQ_API_KEY)

    def _load_model_pool(self):
        logger.info(f"Creating LLM pool with {self.pool_size} instance(s)")
        for i in range(self.pool_size):
            instance = LLMModelInstance(instance_id=i)
            if instance.is_available():
                self.model_instances.append(instance)
                self.instance_queue.put(instance)
            else:
                logger.warning(
                    f"Failed to load instance {i}, pool will have fewer instances"
                )

        if not self.model_instances:
            logger.error("No LLM instances loaded successfully!")

        else:
            logger.info(
                f"LLM pool ready with {len(self.model_instances)}/{self.pool_size} instances. "
            )

    def _get_system_prompt(self) -> str:
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
            • "new line" / "new paragraph" → line break
            • "bullet point" / "dash" → • (when creating lists)
            • "number one, number two" → 1. 2. (when creating numbered lists)

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
            """

    def _create_correction_prompt(self, text: str) -> str:
        return f"{self._get_system_prompt()}\n\nPlease correct this transcribed text: {text}\n\nCorrected:"

    def correct_text(self, text: str) -> str:
        if not self.model_instances or not text.strip():
            return text

        instance: Optional[LLMModelInstance] = None
        try:
            try:
                return self._get_correct_text_from_groq(text)
            except Exception as e:
                logger.error(f"Failed to correct text with Groq: {str(e)}")
                pass
            instance = self.instance_queue.get(timeout=config.LLM_REQUEST_TIMEOUT)
            corrected_text = instance.correct(
                prompt=self._create_correction_prompt(text),
            )
            if (
                not corrected_text
                or len(corrected_text) < len(text) * config.LLM_CORRECTION_THRESHOLD
            ):
                logger.warning(
                    "LLM correction resulted in unusually short text, using original"
                )
                return text

            return corrected_text

        except queue.Empty:
            logger.error(
                f"Failed to acquire LLM instance after {config.LLM_REQUEST_TIMEOUT:.2f}s timeout. "
            )
            return text

        except Exception as e:
            logger.error(f"LLM text correction failed: {str(e)}")
            return text

        finally:
            if instance:
                self.instance_queue.put(instance)

    def _get_correct_text_from_groq(self, text: str) -> str:
        try:
            completion = self.groq.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": text},
                ],
                temperature=1,
                max_completion_tokens=2048,
                top_p=1,
                reasoning_effort="low",
                stream=False,
                stop=None,
            )
            content = completion.choices[0].message.content
            return content.strip()
        except Exception as e:
            logger.error(f"Failed to correct text with Groq: {str(e)}")
            return text

    def warmup(self):
        if not self.model_instances:
            return

        logger.info(f"Warming up {len(self.model_instances)} LLM instance(s)...")
        for instance in self.model_instances:
            if instance.is_available():
                try:
                    inst = self.instance_queue.get(timeout=5.0)
                    inst.correct(
                        prompt="Warm up the model",
                    )
                    self.instance_queue.put(inst)
                    logger.info(f"Instance {instance.instance_id} warmed up")
                except Exception as e:
                    logger.warning(
                        f"Failed to warm up instance {instance.instance_id}: {e}"
                    )
        logger.info("All LLM instances warmed up successfully")

    def is_available(self) -> bool:
        return len(self.model_instances) > 0

# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import yaml
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.endpoints.endpoint_model import ModelInfo
from lighteval.models.model_input import GenerationParameters
from lighteval.models.model_output import (
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
)
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)
from lighteval.utils.imports import is_litellm_available


logger = logging.getLogger(__name__)

if is_litellm_available():
    import litellm
    from litellm import encode
    from litellm.caching.caching import Cache
    from litellm.utils import ModelResponse

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").handlers.clear()

    litellm.cache = Cache(type="disk")


@dataclass
class LiteLLMModelConfig:
    model: str
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    generation_parameters: GenerationParameters = None

    def __post_init__(self):
        if self.generation_parameters is None:
            self.generation_parameters = GenerationParameters()

    @classmethod
    def from_path(cls, path):
        with open(path, "r") as f:
            config = yaml.safe_load(f)["model"]

        model = config["base_params"]["model_name"]
        provider = config["base_params"].get("provider", None)
        base_url = config["base_params"].get("base_url", None)
        api_key = config["base_params"].get("api_key", None)
        generation_parameters = GenerationParameters.from_dict(config)
        return cls(
            model=model,
            provider=provider,
            base_url=base_url,
            generation_parameters=generation_parameters,
            api_key=api_key,
        )


class LiteLLMClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config, env_config) -> None:
        """
        IMPORTANT: Your API keys should be set in the environment variables.
        If a base_url is not set, it will default to the public API.
        """
        self.model_info = ModelInfo(
            model_name=config.model,
            model_sha="",
            model_dtype=None,
            model_size="",
        )
        self.model = config.model
        self.provider = config.provider or config.model.split("/")[0]
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.generation_parameters = config.generation_parameters

        self.API_MAX_RETRY = 5
        self.API_RETRY_SLEEP = 3
        self.API_RETRY_MULTIPLIER = 2
        self.CONCURENT_CALLS = 1  # 100 leads to hitting Anthropic rate limits

        self._tokenizer = encode
        self.pairwise_tokenization = False
        litellm.drop_params = True
        litellm.set_verbose = False

    def _prepare_stop_sequence(self, stop_sequence):
        """Prepare and validate stop sequence."""
        if self.provider == "anthropic":
            # Filter out whitespace-only stop sequences
            if stop_sequence:
                stop_sequence = [s for s in stop_sequence if s and s.strip()]
        return stop_sequence

    def _prepare_max_new_tokens(self, max_new_tokens):
        """Calculate completion tokens based on max_new_tokens."""
        if not max_new_tokens or max_new_tokens <= 0:
            return None

        if "o1" in self.model:
            # We need to allow more tokens to include reasoning tokens
            max_new_tokens = min(max_new_tokens * 10, 32000)
        return max_new_tokens

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence):
        """Make API call with retries."""
        response = None
        for attempt in range(self.API_MAX_RETRY):
            try:
                stop_sequence = self._prepare_stop_sequence(stop_sequence)
                max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)
    
                if return_logits and not self.provider == "openai":
                    logger.warning("Returning logits is not supported for this provider, ignoring.")
    
                # Format the messages parameter correctly based on input type
                if isinstance(prompt, str):
                    # Convert string to proper message format
                    messages = [{"role": "user", "content": prompt}]
                elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
                    # Already in proper format
                    messages = prompt
                else:
                    # Handle unexpected format
                    logger.warning(f"Unexpected prompt format: {type(prompt)}. Converting to user message.")
                    messages = [{"role": "user", "content": str(prompt)}]
    
                # Prepare kwargs for completion call
                kwargs = {
                    "model": self.model,
                    "messages": messages,  # Use the correctly formatted messages
                    "logprobs": return_logits if self.provider == "openai" else None,
                    "base_url": self.base_url,
                    "n": num_samples,
                    # "caching": True,
                    "caching": False,
                    "api_key": self.api_key,
                    "request_timeout": 1000,
                }
                
                if "o1" in self.model:
                    logger.warning("O1 models do not support temperature, top_p, stop sequence. Disabling.")
                else:
                    if hasattr(self, 'generation_parameters') and self.generation_parameters is not None:
                        if hasattr(self.generation_parameters, 'to_litellm_dict'):
                            gen_params = self.generation_parameters.to_litellm_dict()
                            if isinstance(gen_params, dict):
                                kwargs.update(gen_params)
    
                if kwargs.get("max_completion_tokens", None) is None:
                    kwargs["max_completion_tokens"] = max_new_tokens
                    
                # Add stop sequences if provided
                if stop_sequence:
                    kwargs["stop"] = stop_sequence
    
                response = litellm.completion(**kwargs)
    
                # If response is empty, retry without caching
                if response and hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                    if response.choices[0].message.content is None:
                        kwargs["caching"] = False
                        logger.info("Response is empty, retrying without caching")
                        response = litellm.completion(**kwargs)
                return response
                
            except Exception as e:
                wait_time = min(64, self.API_RETRY_SLEEP * (2**attempt))
                logger.warning(
                    f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait_time)
    
        logger.error(f"API call failed after {self.API_MAX_RETRY} attempts, returning empty response.")
        from litellm.utils import ModelResponse
        return ModelResponse()

    def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int],
        num_samples: int | list[int],
        stop_sequence: list[str] | None = None,
    ):
        logger.info(f"=== STARTING PARALLEL API CALLS FOR {len(prompts)} PROMPTS ===")
        results = []
    
        # Prepare parameters
        logger.info("Preparing parameters for parallel calls")
        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        
        # Prepare stop sequences - with detailed logging
        if stop_sequence is None:
            logger.info("stop_sequence is None, creating list of Nones")
            stop_sequencess = [None for _ in prompts]
        else:
            logger.info(f"Creating stop_sequencess from: {stop_sequence}")
            stop_sequencess = [stop_sequence for _ in prompts]
        
        # Log the parameter lengths for debugging
        logger.info(f"prompts length: {len(prompts)}")
        logger.info(f"return_logitss length: {len(return_logitss)}")
        logger.info(f"max_new_tokenss length: {len(max_new_tokenss)}")
        logger.info(f"num_sampless length: {len(num_sampless)}")
        logger.info(f"stop_sequencess length: {len(stop_sequencess)}")
        
        try:
            assert (
                len(prompts) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(stop_sequencess)
            ), f"Length mismatch: {len(prompts)}, {len(return_logitss)}, {len(max_new_tokenss)}, {len(num_sampless)}, {len(stop_sequencess)}"
        except AssertionError as e:
            logger.error(f"Assertion error in parameter lengths: {e}")
            # Try to fix the lengths to avoid failing
            min_len = min(len(prompts), len(return_logitss), len(max_new_tokenss), len(num_sampless), len(stop_sequencess))
            logger.info(f"Truncating all parameters to length {min_len}")
            prompts = prompts[:min_len]
            return_logitss = return_logitss[:min_len]
            max_new_tokenss = max_new_tokenss[:min_len]
            num_sampless = num_sampless[:min_len]
            stop_sequencess = stop_sequencess[:min_len]
    
        logger.info("Starting ThreadPoolExecutor")
        with ThreadPoolExecutor(self.CONCURENT_CALLS) as executor:
            futures = []
            for i in range(len(prompts)):
                logger.info(f"Submitting task {i+1}/{len(prompts)}")
                future = executor.submit(
                    self.__call_api,
                    prompts[i],
                    return_logitss[i],
                    max_new_tokenss[i],
                    num_sampless[i],
                    stop_sequencess[i],
                )
                futures.append(future)
            
            for i, future in enumerate(tqdm(futures, total=len(futures))):
                try:
                    logger.info(f"Getting result for task {i+1}/{len(futures)}")
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in future {i}: {str(e)}")
                    # Add an empty response to maintain the order
                    from litellm.utils import ModelResponse
                    results.append(ModelResponse())
    
        logger.info(f"Parallel execution complete, got {len(results)} results")
        
        # Check for None values in results
        none_count = sum(1 for r in results if r is None)
        if none_count > 0:
            logger.warning(f"Found {none_count} None results - replacing with empty responses")
            from litellm.utils import ModelResponse
            results = [r if r is not None else ModelResponse() for r in results]
    
        return results

    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerativeResponse]: list of generated responses.
        """
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,  # self.disable_tqdm,
        ):
            contexts = [c.context for c in dataset]
            max_new_tokens = dataset[0].generation_size  # could be none
            return_logits = dataset[0].use_logits
            num_samples = dataset[0].num_samples
            stop_sequence = requests[0].stop_sequence

            responses = self.__call_api_parallel(contexts, return_logits, max_new_tokens, num_samples, stop_sequence)

            for response in responses:
                result: list[str] = [choice.message.content for choice in response.choices]

                cur_response = GenerativeResponse(
                    # In empty responses, the model should return an empty string instead of None
                    result=result if result[0] else [""],
                    logits=None,
                    generated_tokens=[],
                    input_tokens=[],
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        return self._tokenizer

    def _encode(self, text: str):
        enc = encode(model=self.model, text=text)
        if hasattr(enc, "ids"):
            return enc.ids
        return enc

    def tok_encode(self, text: str | list[str]):
        if isinstance(text, list):
            toks = [self._encode(t["content"]) for t in text]
            toks = [tok for tok in toks if tok]
            return toks
        return self._encode(text)

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        return 4096

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError

    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

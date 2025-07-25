"""
Local Transformers inference backend
"""

import torch
import numpy as np
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from .base import InferenceBackend, InferenceConfig, EmbeddingResult, GenerationResult

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ImportError as e:
    raise ImportError("transformers library not available") from e

# cuda mem optimizations
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = logging.getLogger(__name__)

@dataclass
class LocalTransformersConfig(InferenceConfig):
    """Configuration for local transformers backend"""
    use_4bit: bool = True
    use_nested_quant: bool = True
    use_cpu_offload: bool = False
    torch_dtype: str = "float16"
    low_cpu_mem_usage: bool = True
    offload_folder: str = "offload_folder"
    trust_remote_code: bool = False

class LocalTransformersBackend(InferenceBackend):
    """Local transformers inference backend"""
    
    def __init__(self, config: LocalTransformersConfig):
        super().__init__(config)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.bnb_config = None
        
    async def load_model(self) -> Dict[str, Any]:
        """Load model with quantization support"""
        start_time = time.time()
        
        # Check for Hymba-specific dependencies
        missing_deps = []
        if "Hymba" in self.config.model_name:
            try:
                # https://github.com/pytorch/pytorch/issues/37377
                # needed for nvidia hymba 1.5
                # Use GNU threading layer to match system's OpenMP (libgomp)
                os.environ["MKL_THREADING_LAYER"] = "GNU"
            except Exception:
                logger.warning(f"MKL_THREADING_LAYER env var not set.")
            try:
                import mamba_ssm
            except ImportError:
                missing_deps.append("mamba_ssm")
            
            try:
                import causal_conv1d
            except ImportError:
                missing_deps.append("causal_conv1d")
                
            try:
                import flash_attn
            except ImportError:
                missing_deps.append("flash_attn")
            
            if missing_deps:
                logger.warning(f"Missing Hymba dependencies: {missing_deps}")
                logger.warning("Model will attempt to load with fallback mode")
        
        # Setup quantization if needed / available
        if self.config.use_4bit and torch.cuda.is_available():
            try:
                self.bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=self.config.use_nested_quant,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.torch_dtype),
                    bnb_4bit_quant_storage=torch.uint8
                )
                logger.info("4-bit quantization enabled")
            except Exception as e:
                logger.warning(f"Could not setup quantization: {e}")
                self.bnb_config = None

        
        logger.info(f"Loading tokenizer for {self.config.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
        except Exception as e:
            if missing_deps:
                logger.error(f"Failed to load tokenizer, likely due to missing dependencies: {missing_deps}")
                logger.info("Install missing dependencies with: pip install mamba_ssm causal_conv1d flash_attn")
            raise e
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto" if self.config.use_cpu_offload else self.device
        
        logger.info(f"Loading model {self.config.model_name}")
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": getattr(torch, self.config.torch_dtype),
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "trust_remote_code": self.config.trust_remote_code
        }
        
        if self.bnb_config is not None:
            model_kwargs["quantization_config"] = self.bnb_config
            
        if self.config.use_cpu_offload:
            model_kwargs["offload_folder"] = self.config.offload_folder
            model_kwargs["offload_state_dict"] = True
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
        except Exception as e:
            if missing_deps and "Hymba" in self.config.model_name:
                logger.error(f"Failed to load Hymba model, likely due to missing dependencies: {missing_deps}")
                logger.info("Try installing with compatible CUDA version:")
                logger.info("pip install mamba_ssm causal_conv1d flash_attn --extra-index-url https://pypi.nvidia.com")
            raise e
        
        self.is_loaded = True
        load_time = time.time() - start_time
        
        try:
            model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        except:
            model_size = 0
        
        stats = {
            "backend": self.backend_name,
            "load_time": load_time,
            "model_size_params": model_size,
            "device": str(self.device),
            "quantization_enabled": self.bnb_config is not None,
            "cpu_offload_enabled": self.config.use_cpu_offload,
            "torch_dtype": self.config.torch_dtype
        }
        
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
        return stats
    
    async def generate_embeddings(self, texts: List[str], **kwargs) -> EmbeddingResult:
        """Generate embeddings using local model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        batch_size = kwargs.get('batch_size', self.config.batch_size)
        max_length = kwargs.get('max_length', self.config.max_length)
        chunk_size = kwargs.get('chunk_size', 200)
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        self.model.eval()
        all_embeddings = []
        
        for chunk_start in range(0, len(texts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]
            
            chunk_embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]
                
                # For Hymba models, process texts one by one to avoid tensor mismatch
                if "Hymba" in self.config.model_name:
                    batch_embeddings_list = []
                    for text in batch_texts:
                        try:
                            with torch.no_grad():
                                inputs = self.tokenizer(
                                    [text],
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=max_length,
                                    padding=False  # No padding for single text
                                )
                                # Ensure input_ids are long tensors (required for embedding layer)
                                if 'input_ids' in inputs and inputs['input_ids'].dtype != torch.long:
                                    inputs['input_ids'] = inputs['input_ids'].long()
                                inputs = inputs.to(self.device)
                                
                                outputs = self.model(**inputs, output_hidden_states=True)
                                last_hidden = outputs.hidden_states[-1]
                                # Convert to float32 first to avoid BFloat16 issues
                                if last_hidden.dtype == torch.bfloat16:
                                    last_hidden = last_hidden.to(torch.float32)
                                single_embedding = last_hidden.mean(dim=1).cpu().numpy()
                                batch_embeddings_list.append(single_embedding)
                                
                                del inputs, outputs, last_hidden
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                        except Exception as e:
                            logger.error(f"Error processing single Hymba text: {e}")
                            dim = getattr(self.model.config, 'hidden_size', 1600)
                            batch_embeddings_list.append(np.random.randn(1, dim).astype(np.float64))
                    
                    if batch_embeddings_list:
                        batch_embeddings = np.concatenate(batch_embeddings_list, axis=0)
                    else:
                        dim = getattr(self.model.config, 'hidden_size', 1600)
                        batch_embeddings = np.random.randn(len(batch_texts), dim).astype(np.float64)
                    
                    # check for NaN values in embeddings
                    if np.isnan(batch_embeddings).any():
                        logger.warning(f"Found NaN in batch embeddings, using random fallback")
                        dim = getattr(self.model.config, 'hidden_size', 1600)
                        batch_size_actual = len(batch_texts)
                        batch_embeddings = np.random.randn(batch_size_actual, dim).astype(np.float64)
                    
                    chunk_embeddings.append(batch_embeddings)
                else:
                    # Standard batch processing for non-Hymba models
                    try:
                        with torch.no_grad():
                            inputs = self.tokenizer(
                                batch_texts,
                                return_tensors="pt",
                                truncation=True,
                                max_length=max_length,
                                padding=True
                            )
                            # Ensure input_ids are long tensors (required for embedding layer)
                            if 'input_ids' in inputs and inputs['input_ids'].dtype != torch.long:
                                inputs['input_ids'] = inputs['input_ids'].long()
                            inputs = inputs.to(self.device)
                            
                            outputs = self.model(**inputs, output_hidden_states=True)
                            last_hidden = outputs.hidden_states[-1]
                            # Convert to float32 first to avoid BFloat16 issues
                            if last_hidden.dtype == torch.bfloat16:
                                last_hidden = last_hidden.to(torch.float32)
                            batch_embeddings = last_hidden.mean(dim=1).cpu().numpy()
                        
                        # check for NaN values in embeddings
                        if np.isnan(batch_embeddings).any():
                            logger.warning(f"Found NaN in batch embeddings, using random fallback")
                            dim = getattr(self.model.config, 'hidden_size', 768)
                            batch_size_actual = len(batch_texts)
                            # Use float64 to avoid JSON serialization issues, will be converted later
                            batch_embeddings = np.random.randn(batch_size_actual, dim).astype(np.float64)
                        
                        chunk_embeddings.append(batch_embeddings)
                        
                        # Mem cleanup
                        del inputs, outputs, last_hidden
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                    except Exception as e:
                        logger.error(f"Error processing batch {i}: {e}")
                        # Fallback: process one by one
                        for text in batch_texts:
                            try:
                                with torch.no_grad():
                                    inputs = self.tokenizer([text], return_tensors="pt",
                                                          truncation=True, max_length=max_length)
                                    # Ensure input_ids are long tensors (required for embedding layer)
                                    if 'input_ids' in inputs and inputs['input_ids'].dtype != torch.long:
                                        inputs['input_ids'] = inputs['input_ids'].long()
                                    inputs = inputs.to(self.device)
                                    outputs = self.model(**inputs, output_hidden_states=True)
                                    last_hidden = outputs.hidden_states[-1]
                                    # Convert to float32 first to avoid BFloat16 issues
                                    if last_hidden.dtype == torch.bfloat16:
                                        last_hidden = last_hidden.to(torch.float32)
                                    single_emb = last_hidden.mean(dim=1).cpu().numpy()

                                    if np.isnan(single_emb).any():
                                        logger.warning(f"Found NaN in single text embedding, using random fallback")
                                        dim = getattr(self.model.config, 'hidden_size', 768)
                                        # float64 to avoid JSON serialization issues, will be converted later
                                        single_emb = np.random.randn(1, dim).astype(np.float64)
                                    
                                    chunk_embeddings.append(single_emb)
                                    
                                    del inputs, outputs, last_hidden
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                            except Exception as e2:
                                logger.error(f"Error processing single text: {e2}")
                                # Last resort: random embedding
                                dim = getattr(self.model.config, 'hidden_size', 768)
                                # Use float64 to avoid JSON serialization issues, will be converted later  
                                chunk_embeddings.append(np.random.randn(1, dim).astype(np.float64))
            
            if chunk_embeddings:
                chunk_array = np.concatenate(chunk_embeddings, axis=0)
                all_embeddings.append(chunk_array)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Processed chunk {chunk_start}-{chunk_end} / {len(texts)}")
        
        if all_embeddings:
            embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            # Fallback empty embeddings
            dim = getattr(self.model.config, 'hidden_size', 768)
            embeddings = np.zeros((len(texts), dim), dtype=np.float64)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Generated embeddings shape {embeddings.shape} in {processing_time:.2f}s")
        
        return EmbeddingResult(
            embeddings=embeddings,
            metadata={
                "backend": self.backend_name,
                "batch_size": batch_size,
                "max_length": max_length,
                "chunk_size": chunk_size,
                "samples_processed": len(texts),
                "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0
            },
            processing_time=processing_time
        )
    
    async def generate_text(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text completion"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.7)
        do_sample = kwargs.get('do_sample', True)
        use_thinking = kwargs.get('use_thinking', False)  # New parameter for thinking control
        
        # Filter out kwargs that we handle explicitly to avoid duplication
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['max_new_tokens', 'temperature', 'do_sample', 'use_thinking']}
        
        logger.debug(f"Generating text for prompt: {prompt[:50]}...")
        
        # Check for model-specific handling
        is_deepseek = "deepseek" in self.config.model_name.lower()
        is_hymba = "hymba" in self.config.model_name.lower()
        is_granite = "granite" in self.config.model_name.lower()
        is_llama = "llama" in self.config.model_name.lower()
        is_smollm = "smollm" in self.config.model_name.lower()
        is_mistral = "mistral" in self.config.model_name.lower()
        
        # Handle model-specific prompt formatting
        generation_prompt = prompt
        model_specific_kwargs = {}
        
        if is_deepseek:
            # For DeepSeek models, always prepend "<think>\n" (they are reasoning models by nature)
            generation_prompt = f"<think>\n{prompt}"
            logger.debug(f"Applied DeepSeek thinking pattern (always enabled)")
        elif is_granite and use_thinking:
            # For Granite models, use chat template with thinking=True only when thinking is requested
            try:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    # Use thinking=True for Granite models following official docs
                    # This returns pre-tokenized input that we'll use directly in generation
                    granite_inputs = self.tokenizer.apply_chat_template(
                        messages, 
                        return_tensors="pt", 
                        thinking=True, 
                        return_dict=True, 
                        add_generation_prompt=True
                    ).to(self.device)
                    
                    # Store the special inputs for later use in generation
                    model_specific_kwargs['granite_inputs'] = granite_inputs
                    generation_prompt = None  # Signal to use granite_inputs instead
                    logger.debug(f"Applied Granite chat template with thinking=True")
                else:
                    # Fallback if apply_chat_template not available
                    logger.warning("Granite chat template not available - using standard format")
                    generation_prompt = prompt
            except Exception as e:
                logger.warning(f"Failed to apply Granite thinking template: {e}, using fallback")
                generation_prompt = prompt
        elif is_granite:
            # For Granite models without thinking, use standard chat template
            try:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    generation_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    logger.debug(f"Applied Granite standard chat template")
                else:
                    # Fallback if apply_chat_template not available
                    logger.warning("Granite chat template not available - using standard format")
                    generation_prompt = prompt
                
                # Granite works well with default parameters, no overrides needed
                
            except Exception as e:
                logger.warning(f"Failed to apply Granite chat template: {e}, using fallback")
                generation_prompt = prompt
        elif is_hymba:
            # For Hymba models, use chat template format
            try:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    generation_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    logger.debug(f"Applied Hymba chat template")
                else:
                    # Fallback manual format if apply_chat_template not available
                    generation_prompt = f"<extra_id_1>User\n{prompt}\n<extra_id_1>Assistant\n"
                    logger.warning("Using manual Hymba format - apply_chat_template not available")
                
                # Override generation parameters for Hymba (following HuggingFace recommendations)
                model_specific_kwargs.update({
                    'do_sample': False,
                    'temperature': 0.7,
                    'max_new_tokens': min(max_new_tokens, 256)  # Hymba recommendation: max 256
                })
                logger.debug(f"Applied Hymba-specific generation parameters: {model_specific_kwargs}")
                
            except Exception as e:
                logger.warning(f"Failed to apply Hymba chat template: {e}, using fallback")
                generation_prompt = prompt
        elif is_llama:
            # For Llama models, use standard chat template format
            try:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    generation_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    logger.debug(f"Applied Llama chat template")
                else:
                    # Fallback if apply_chat_template not available
                    logger.warning("Llama chat template not available - using standard format")
                    generation_prompt = prompt
                
                # Llama works well with default parameters, no specific overrides needed
                
            except Exception as e:
                logger.warning(f"Failed to apply Llama chat template: {e}, using fallback")
                generation_prompt = prompt
        elif is_smollm:
            # For SmolLM2 models, use standard chat template with recommended parameters
            try:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    generation_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    logger.debug(f"Applied SmolLM2 chat template")
                else:
                    # Fallback if apply_chat_template not available
                    logger.warning("SmolLM2 chat template not available - using standard format")
                    generation_prompt = prompt
                
                # Apply SmolLM2 recommended generation parameters
                model_specific_kwargs.update({
                    'temperature': 0.2,  # SmolLM2 recommendations
                    'top_p': 0.9,    
                    'do_sample': True,
                    'max_new_tokens': min(max_new_tokens, 512)  # limit for small model
                })
                logger.debug(f"Applied SmolLM2-specific generation parameters: {model_specific_kwargs}")
                
            except Exception as e:
                logger.warning(f"Failed to apply SmolLM2 chat template: {e}, using fallback")
                generation_prompt = prompt
        elif is_mistral:
            try:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    generation_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    logger.debug(f"Applied Mistral chat template")
                else:
                    # Fallback if apply_chat_template not available
                    logger.warning("Mistral chat template not available - using standard format")
                    generation_prompt = prompt
            except Exception as e:
                logger.warning(f"Failed to apply Mistral chat template: {e}, using fallback")
                generation_prompt = prompt
        
        # ... input preparation based on model type
        if is_granite and 'granite_inputs' in model_specific_kwargs:
            # Use pre-tokenized Granite inputs
            inputs = model_specific_kwargs.pop('granite_inputs')  # Remove from kwargs to avoid duplication
        else:
            # Standard tokenization for other models
            inputs = self.tokenizer(generation_prompt, return_tensors="pt")
            # Ensure input_ids are long tensors (required for embedding layer)
            if 'input_ids' in inputs and inputs['input_ids'].dtype != torch.long:
                inputs['input_ids'] = inputs['input_ids'].long()
            inputs = inputs.to(self.device)
        
        try:
            with torch.no_grad():
                # Merge model-specific kwargs with user kwargs (model-specific takes precedence)
                generation_kwargs = {
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'do_sample': do_sample,
                    'pad_token_id': self.tokenizer.eos_token_id,
                    **filtered_kwargs,
                    **model_specific_kwargs  # Model-specific overrides
                }
                
                # Add stop tokens for Hymba
                if is_hymba:
                    # Hymba uses </s> as stop token
                    eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.convert_tokens_to_ids("</s>")
                    if eos_token_id is not None:
                        generation_kwargs['eos_token_id'] = eos_token_id
                
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate the generated portion based on input type
            if is_granite and generation_prompt is None:
                # For Granite with chat template, use input_ids length for slicing
                input_length = inputs["input_ids"].shape[1]
                generated_text = self.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()
            else:
                # Standard approach for other models
                generated_text = response[len(generation_prompt):].strip()
            
            # Model-specific post-processing
            if is_deepseek and not generated_text.startswith("<think>"):
                # include the thinking tokens in the output (DeepSeek always thinks)
                generated_text = f"<think>\n{generated_text}"
            elif is_granite:
                # Granite with thinking mode, response should already be properly formatted
                pass
            elif is_llama:
                # no special post-processing needed
                pass
            elif is_smollm:
                # no special post-processing needed
                pass
            elif is_mistral:
                # no special post-processing needed 
                pass
            elif is_hymba:
                # post: clean up any remaining special tokens
                generated_text = generated_text.replace("<extra_id_1>", "").replace("<extra_id_0>", "")
                generated_text = generated_text.replace("Assistant", "").strip()
                generated_text = generated_text.strip()            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            generated_text = f"[Error: {str(e)}]"
        
        processing_time = time.time() - start_time
        
        return GenerationResult(
            text=generated_text,
            metadata={
                "backend": self.backend_name,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "prompt_length": len(prompt)
            },
            processing_time=processing_time
        )
    
    async def unload_model(self):
        """Clean up model resources with aggressive GPU memory cleanup"""
        logger.info("Unloading model resources")
        
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Aggressive GPU memory cleanup
        if torch.cuda.is_available():
            # Multi-pass cleanup with validation
            max_retries = 3
            for attempt in range(max_retries):
                # Clear all GPU caches
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Clear cache for all devices
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                # Check memory state
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                
                logger.info(f"GPU Memory after cleanup (attempt {attempt+1}): "
                           f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
                
                # If memory is sufficiently freed, break
                if allocated < 0.1:  # Less than 100MB allocated
                    break
                    
                if attempt < max_retries - 1:
                    logger.warning(f"Memory not fully freed, retrying cleanup...")
                    import time
                    time.sleep(2)
                    gc.collect()
        
        self.is_loaded = False
        logger.info("Model unloaded successfully")
    
    @property
    def backend_name(self) -> str:
        return "local_transformers"
    
    @property
    def supports_batch_inference(self) -> bool:
        return True
    
    @property
    def max_batch_size(self) -> int:
        return 32  # Conservative default, can be configured
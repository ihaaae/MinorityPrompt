from pathlib import Path
from typing import Any, Dict, Optional

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from torch.optim.adam import Adam
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


__SOLVER__ = {}

MODEL_CACHE_ROOT = Path(__file__).resolve().parent / "models" / "huggingface"
SD3_REPO = "stabilityai/stable-diffusion-3-medium-diffusers"
SD35_REPO = "stabilityai/stable-diffusion-3.5-medium"


def model_cache_dir() -> str:
    return str(MODEL_CACHE_ROOT)


def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls

    return wrapper


def get_solver(name: str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)


class SD3:
    def __init__(
        self,
        solver_config: Dict,
        model_key: str = SD3_REPO,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        reuse_sd3_text: bool = False,
        **kwargs,
    ):
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir or model_cache_dir()
        self.num_sampling = solver_config.num_sampling

        pipe_kwargs = {
            "torch_dtype": dtype,
            "cache_dir": self.cache_dir,
        }
        if reuse_sd3_text:
            pipe_kwargs.update(self._sd3_text_components(dtype))
        else:
            pipe_kwargs["tokenizer_3"] = T5TokenizerFast.from_pretrained(
                model_key,
                subfolder="tokenizer_3",
                cache_dir=self.cache_dir,
                from_slow=True,
            )

        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_key, **pipe_kwargs).to(device)
        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

        self.vae_scale_factor = self.pipe.vae_scale_factor
        self.default_sample_size = self.pipe.default_sample_size

    def _sd3_text_components(self, dtype: torch.dtype):
        return {
            "tokenizer": CLIPTokenizer.from_pretrained(SD3_REPO, subfolder="tokenizer", cache_dir=self.cache_dir),
            "tokenizer_2": CLIPTokenizer.from_pretrained(SD3_REPO, subfolder="tokenizer_2", cache_dir=self.cache_dir),
            "tokenizer_3": T5TokenizerFast.from_pretrained(
                SD3_REPO,
                subfolder="tokenizer_3",
                cache_dir=self.cache_dir,
                from_slow=True,
            ),
            "text_encoder": CLIPTextModelWithProjection.from_pretrained(
                SD3_REPO,
                subfolder="text_encoder",
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
            ),
            "text_encoder_2": CLIPTextModelWithProjection.from_pretrained(
                SD3_REPO,
                subfolder="text_encoder_2",
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
            ),
            "text_encoder_3": T5EncoderModel.from_pretrained(
                SD3_REPO,
                subfolder="text_encoder_3",
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
            ),
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.sample(*args, **kwargs)

    def encode_prompt(self, prompt, negative_prompt=""):
        return self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
            do_classifier_free_guidance=True,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=256,
        )

    def predict_flow(self, zt, t, prompt_embeds, pooled_prompt_embeds):
        timestep = t.expand(zt.shape[0]) if len(t.shape) == 0 else t
        return self.transformer(
            hidden_states=zt,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds.to(dtype=self.dtype),
            pooled_projections=pooled_prompt_embeds.to(dtype=self.dtype),
            return_dict=False,
        )[0]

    def initialize_latent(self, height=1024, width=1024, generator=None):
        shape = (
            1,
            self.transformer.config.in_channels,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        return randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype).requires_grad_()

    def decode(self, zt):
        latents = (zt / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(latents, return_dict=False)[0]

    def restore_embedding(self, placeholder_token_ids, orig_embeds_params, tokenizer, text_enc):
        index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool, device=orig_embeds_params.device)
        index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

        with torch.no_grad():
            text_enc.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]

    def initialize_embedding(self, tokenizer, text_enc, popt_kwargs):
        num_opt_tokens = popt_kwargs["num_opt_tokens"]
        init_type = popt_kwargs["init_type"]
        init_word = popt_kwargs["init_word"]
        placeholder_string = popt_kwargs["placeholder_string"]
        assert init_type == "word"
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2

        placeholder_symbol = placeholder_string.split("_")[0]
        placeholder_tokens = [placeholder_string]
        for i in range(1, num_opt_tokens):
            placeholder_tokens.append(f"{placeholder_symbol}_{i}")

        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != num_opt_tokens:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_string}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        text_enc.resize_token_embeddings(len(tokenizer))

        token_ids = tokenizer.encode(init_word, add_special_tokens=False)
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        token_embeds = text_enc.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id in placeholder_token_ids:
                token_embeds[token_id] = token_embeds[token_ids[0]].clone()

        text_enc.text_model.encoder.requires_grad_(False)
        text_enc.text_model.final_layer_norm.requires_grad_(False)
        text_enc.text_model.embeddings.position_embedding.requires_grad_(False)
        return placeholder_token_ids

    def prompt_with_placeholder(self, prompt, popt_kwargs):
        placeholder_symbol = popt_kwargs["placeholder_string"].split("_")[0]
        num_opt_tokens = popt_kwargs["num_opt_tokens"]
        placeholder = " ".join(f"{placeholder_symbol}_{i}" for i in range(num_opt_tokens))
        if popt_kwargs["placeholder_position"] == "start":
            return f"{placeholder} {prompt}"
        return f"{prompt} {placeholder}"

    @torch.enable_grad()
    def prompt_opt(self, zt, t, step, placeholder_token_ids, prompt, base_embeds, popt_kwargs):
        decay_rate = popt_kwargs["lr_decay_rate"]
        optimizer = Adam(
            self.text_encoder.get_input_embeddings().parameters(),
            lr=popt_kwargs["p_opt_lr"] * (1.0 - step * decay_rate),
        )
        orig_embeds_params = self.text_encoder.get_input_embeddings().weight.data.clone()
        opt_prompt = self.prompt_with_placeholder(prompt, popt_kwargs)
        base_prompt_embeds, base_pooled_prompt_embeds = base_embeds

        for i in range(popt_kwargs["p_opt_iter"]):
            prompt_embeds, _, pooled_prompt_embeds, _ = self.encode_prompt(opt_prompt)
            flow = self.predict_flow(zt, t, prompt_embeds, pooled_prompt_embeds)

            with torch.no_grad():
                base_flow = self.predict_flow(
                    zt,
                    t,
                    base_prompt_embeds.detach(),
                    base_pooled_prompt_embeds.detach(),
                )

            loss = -1 * (flow - base_flow).reshape(flow.shape[0], -1).norm(p=2.0, dim=-1).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.restore_embedding(placeholder_token_ids, orig_embeds_params, self.tokenizer, self.text_encoder)

        with torch.no_grad():
            prompt_embeds, _, pooled_prompt_embeds, _ = self.encode_prompt(opt_prompt)
        return prompt_embeds, pooled_prompt_embeds

    def get_timesteps(self, latents, num_inference_steps):
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None):
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (width // self.transformer.config.patch_size)
            scheduler_kwargs["mu"] = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            self.device,
            **scheduler_kwargs,
        )
        return timesteps


@register_solver("flowmatch")
class BaseFlowMatch(SD3):
    def sample(
        self,
        cfg_guidance=7.0,
        prompt=["", ""],
        callback_fn=None,
        popt_kwargs=None,
        height=1024,
        width=1024,
        generator=None,
        **kwargs,
    ):
        null_prompt, text_prompt = prompt

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            text_prompt,
            negative_prompt=null_prompt,
        )
        base_embeds = (prompt_embeds.detach().clone(), pooled_prompt_embeds.detach().clone())

        zt = self.initialize_latent(height=height, width=width, generator=generator)
        timesteps = self.get_timesteps(zt, self.num_sampling)

        if popt_kwargs["prompt_opt"]:
            self.text_encoder = self.text_encoder.to(torch.float32)
            placeholder_token_ids = self.initialize_embedding(self.tokenizer, self.text_encoder, popt_kwargs)
            self.vae.requires_grad_(False)
            self.transformer.requires_grad_(False)
            self.pipe.text_encoder_2.requires_grad_(False)
            if self.pipe.text_encoder_3 is not None:
                self.pipe.text_encoder_3.requires_grad_(False)
        else:
            placeholder_token_ids = None

        pbar = tqdm(timesteps, desc="SD3")
        for step, t in enumerate(pbar):
            if popt_kwargs["prompt_opt"] and step < int(len(timesteps) * (1.0 - popt_kwargs["t_lo"])) and step % popt_kwargs["inter_rate"] == 0:
                prompt_embeds, pooled_prompt_embeds = self.prompt_opt(
                    zt.detach(),
                    t,
                    step,
                    placeholder_token_ids,
                    text_prompt,
                    base_embeds,
                    popt_kwargs,
                )
            elif popt_kwargs["prompt_opt"] and popt_kwargs["base_prompt_after_popt"]:
                prompt_embeds, pooled_prompt_embeds = base_embeds

            with torch.no_grad():
                latent_model_input = torch.cat([zt] * 2)
                timestep = t.expand(latent_model_input.shape[0])
                cond_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(dtype=self.dtype)
                pooled_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0).to(dtype=self.dtype)
                flow_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=cond_embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]
                flow_uncond, flow_text = flow_pred.chunk(2)
                flow_pred = flow_uncond + cfg_guidance * (flow_text - flow_uncond)
                zt = self.scheduler.step(flow_pred, t, zt, return_dict=False)[0]

        return self.decode(zt)

from causvid.models.model_interface import (
    DiffusionModelInterface,
    TextEncoderInterface,
    VAEInterface
)
from causvid.models.wan.wan_base.modules.tokenizers import HuggingfaceTokenizer
from causvid.models.wan.wan_base.modules.model import WanModel
from causvid.models.wan.wan_base.modules.vae import _video_vae
from causvid.models.wan.wan_base.modules.t5 import umt5_xxl
from causvid.models.wan.flow_match import FlowMatchScheduler
from causvid.models.wan.causal_model import CausalWanModel
from typing import List, Tuple, Dict, Optional
import torch
import os
import torch.distributed as dist
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class WanTextEncoder(TextEncoderInterface):
    def __init__(self, model_type="T2V-1.3B") -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(
                os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/models_t5_umt5-xxl-enc-bf16.pth"),
                map_location='cpu', weights_only=False
            )
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/google/umt5-xxl/"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(VAEInterface):
    def __init__(self, model_type="T2V-1.3B"):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path=os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False)

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.decode(u.unsqueeze(0),
                              scale).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = self.model.decode(zs, scale).clamp_(-1, 1)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        # output = output.permute(0, 2, 1, 3, 4)
        return output
    
    def stream_encode(self, video: torch.Tensor, is_scale=False) -> torch.Tensor:
        if is_scale:
            device, dtype = video.device, video.dtype
            scale = [self.mean.to(device=device, dtype=dtype),
                    1.0 / self.std.to(device=device, dtype=dtype)]
        else:
            scale = None
        return self.model.stream_encode(video, scale)
    
    def stream_decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        zs = zs.to(torch.bfloat16).to('cuda')
        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = self.model.stream_decode(zs, scale).float().clamp_(-1, 1)
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(DiffusionModelInterface):
    def __init__(self, model_type="T2V-1.3B"):
        super().__init__()

        self.model = WanModel.from_pretrained(os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/"))
        self.model.eval()

        self.uniform_timestep = True

        self.scheduler = FlowMatchScheduler(
            shift=8.0, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        super().post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return pred_x0

    def forward_input(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor,block_mode: str='input', block_num = None, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
        patched_x_shape: torch.Tensor = None,
        block_x: torch.Tensor = None,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache must be provided"

        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep
        
        if block_x is not None and block_mode == 'middle':
            noisy_image_or_video = block_x
        else:
            noisy_image_or_video = noisy_image_or_video.permute(0, 2, 1, 3, 4)

        output, patched_x_shape = self.model(
            noisy_image_or_video,
            t=input_timestep, context=prompt_embeds,
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            current_end=current_end,
            block_mode=block_mode,
            block_num=block_num,
            patched_x_shape=patched_x_shape,
        )

        return output, patched_x_shape

    def forward_output(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, block_mode: str='output', block_num = None, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
        patched_x_shape: torch.Tensor = None,
        block_x: torch.Tensor = None,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache must be provided"

        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        flow_pred = self.model(
            block_x,
            t=input_timestep, context=prompt_embeds,
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            current_end=current_end,
            block_mode=block_mode,
            block_num=block_num,
            patched_x_shape=patched_x_shape,
        ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return pred_x0


class CausalWanDiffusionWrapper(WanDiffusionWrapper):
    def __init__(self, model_type="T2V-1.3B"):
        super().__init__()

        self.model = CausalWanModel.from_pretrained(
            os.path.join(repo_root, f"wan_models/Wan2.1-{model_type}/"))
        self.model.eval()

        self.uniform_timestep = False


# --------------------------------------------------------------------------------------------
# TAEHV decoder (plan.md Step 2). Encoder stays Wan; only the decoder is swapped.
# --------------------------------------------------------------------------------------------

TAEHV_DIR = os.environ.get("TAEHV_DIR", "/home/joshua/taehv")
TAEHV_CKPT = os.environ.get("TAEHV_CKPT", "taew2_1.pth")


def _import_taehv(taehv_dir: str):
    """Import taehv.py either as an installed package or by file path from `taehv_dir`."""
    try:
        import taehv  # noqa: F401  (pip install -e /home/joshua/taehv)
        return taehv
    except ImportError:
        pass
    import importlib.util
    import sys
    path = os.path.join(taehv_dir, "taehv.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"taehv.py not found at {path}; set TAEHV_DIR or pip install taehv")
    spec = importlib.util.spec_from_file_location("taehv", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["taehv"] = mod
    spec.loader.exec_module(mod)
    return mod


class TAEHVDecoderWrapper(WanVAEWrapper):
    """
    WanVAEWrapper with the decoder replaced by TAEHV (taew2_1, Wan 2.1 latent space).

    Contract (identical to WanVAEWrapper.stream_decode_to_pixel):
        input  latent  [B, T, 16, H/8, W/8]   in the DiT's (Wan-normalized) latent space
        output pixels  [B, T_px, 3, H, W]     float32 in [-1, 1]
    Frame accounting matches the Wan streaming decoder: the first call with T latents returns
    4*T - 3 frames (startup frames trimmed once), every later call returns 4 frames per latent.

    Latent convention: TAEHV consumes exactly what the diffusion model uses (no mean/std
    scaling, see taehv README "How do I use TAEHV with Diffusers"), so the DiT output is fed
    as-is. Verified against a real Wan latent in examples/check_taehv_latent_space.py.

    decode_mode:
        "stream"   StreamingTAEHV.decode per latent, MemBlock state carried across calls (correct).
        "parallel" TAEHV.decode_video(parallel=True) per call: state reset every call and 3 raw
                   frames trimmed per call. Reproduces the time-parallel misuse (hypothesis H3);
                   returns 1 frame per chunk instead of 4.
    The Wan VAE is still loaded (super().__init__) because rank 0 / the single-GPU path call
    stream_encode on the same object.
    """
    decode_mode = "stream"

    def __init__(self, model_type="T2V-1.3B", taehv_dir: Optional[str] = None,
                 taehv_ckpt: Optional[str] = None, decode_mode: Optional[str] = None):
        super().__init__(model_type=model_type)
        self.decode_mode = decode_mode or self.decode_mode
        assert self.decode_mode in ("stream", "parallel"), self.decode_mode
        taehv_dir = taehv_dir or TAEHV_DIR
        ckpt = taehv_ckpt or TAEHV_CKPT
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(taehv_dir, ckpt)
        self._taehv_mod = _import_taehv(taehv_dir)
        self.taehv = self._taehv_mod.TAEHV(checkpoint_path=ckpt).eval().requires_grad_(False)
        assert self.taehv.latent_channels == 16, "taew2_1 expected (16 latent channels)"
        self.streaming = self._taehv_mod.StreamingTAEHV(self.taehv)
        self.taehv_ckpt = ckpt
        self.reset()

    def reset(self):
        """Start a new stream: drop TAEHV MemBlock memory and pending work; reset counters."""
        self.streaming.reset()
        self.num_decode_calls = 0
        self.num_latents_in = 0
        self.num_frames_out = 0

    def _to_taehv(self, latent: torch.Tensor) -> torch.Tensor:
        # [B, T, C, h, w] is already NTCHW, which is TAEHV's layout. Match TAEHV's device/dtype.
        p = next(self.taehv.parameters())
        return latent.to(device=p.device, dtype=p.dtype)

    @staticmethod
    def _from_taehv(frames: torch.Tensor) -> torch.Tensor:
        # NTCHW in [0, 1]  ->  [B, T_px, 3, H, W] float32 in [-1, 1]
        return frames.float().mul_(2.0).sub_(1.0).clamp_(-1.0, 1.0)

    @torch.no_grad()
    def stream_decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        x = self._to_taehv(latent)
        if self.decode_mode == "parallel":
            out = self.taehv.decode_video(x, parallel=True, show_progress_bar=False)
        else:
            frames = []
            f = self.streaming.decode(x)
            while f is not None:
                frames.append(f)
                f = self.streaming.decode()
            if frames:
                out = torch.cat(frames, 1)
            else:  # only possible while startup frames are being consumed (T < 1 latent)
                out = x.new_zeros(x.shape[0], 0, 3, x.shape[3] * 8, x.shape[4] * 8)
        self.num_decode_calls += 1
        self.num_latents_in += x.shape[1]
        self.num_frames_out += out.shape[1]
        return self._from_taehv(out)

    @torch.no_grad()
    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """Stateless full-clip decode (Mode A once over the whole clip); does not touch stream state."""
        out = self.taehv.decode_video(self._to_taehv(latent), parallel=True, show_progress_bar=False)
        return self._from_taehv(out)


class TAEHVParallelDecoderWrapper(TAEHVDecoderWrapper):
    """`--vae taehv_parallel`: the per-chunk time-parallel misuse (H3)."""
    decode_mode = "parallel"


class TAEHVFullWrapper(TAEHVDecoderWrapper):
    """
    `--vae taehv_full` (plan.md Step 5): TAEHV encoder on rank 0 as well as the TAEHV decoder.

    stream_encode contract (identical to WanVAEWrapper.stream_encode as V2 calls it, is_scale=False):
        input  video  [B, 3, T, H, W] in [-1, 1]; first call T = 5, later calls T = 4
        output latent [B, 16, T_l, H/8, W/8]; first call T_l = 2, later T_l = 1

    Latent space: Wan's stream_encode returns the raw posterior mean mu (no mean/std normalization)
    and V2 feeds that to the DiT, whereas TAEHV's encoder produces latents in the normalized space
    the DiT *outputs* (Step 2). encode_space="raw" (default) maps TAEHV's latent back with
    z * std + mean so the DiT sees the same input statistics as today; "norm" feeds it as-is.
    Override with TAEHV_ENC_SPACE=raw|norm.

    Temporal alignment (verified in examples/check_taehv_encoder.py): TAEHV's TPool groups frames
    [4m, 4m+3] into latent m, and that latent matches Wan's latent m (which Wan builds from frames
    [4m-3, 4m]) to ~5 % relative error; feeding it Wan's own grouping instead produces a 3-frame
    temporal offset (12 dB worse). So the streaming encoder is simply fed the frames as they come
    and emits a latent whenever four have accumulated: V2's first call (frames 0..4) yields ONE
    latent (frames 0..3) and holds frame 4; every later 4-frame chunk [4c+1, 4c+4] completes the
    group [4c, 4c+3] and yields latent c. The stream is therefore one latent (4 frames) behind
    Wan's, the first call returns 1 latent instead of 2, and the last partial group is dropped at
    the end of the clip. The inference scripts size the first KV-cache block from the returned
    latent count, so no padding or slicing change is needed.
    """
    encode_space = "raw"

    def __init__(self, model_type="T2V-1.3B", encode_space: Optional[str] = None, **kw):
        super().__init__(model_type=model_type, **kw)
        self.encode_space = encode_space or os.environ.get("TAEHV_ENC_SPACE", self.encode_space)
        assert self.encode_space in ("raw", "norm"), self.encode_space
        self.enc_streaming = self._taehv_mod.StreamingTAEHV(self.taehv)
        self.num_encode_calls = 0
        self.num_frames_in = 0

    def reset(self):
        super().reset()
        if hasattr(self, "enc_streaming"):
            self.enc_streaming.reset()
        self.num_encode_calls = 0
        self.num_frames_in = 0

    @torch.no_grad()
    def stream_encode(self, video: torch.Tensor, is_scale: bool = False) -> torch.Tensor:
        p = next(self.taehv.parameters())
        x = video.permute(0, 2, 1, 3, 4).to(device=p.device, dtype=p.dtype)      # [B, T, 3, H, W]
        x = (x * 0.5 + 0.5).clamp_(0.0, 1.0)                                    # TAEHV wants [0, 1]
        self.num_frames_in += x.shape[1]
        lats = []
        z = self.enc_streaming.encode(x)          # frames are queued; a latent pops out per 4 accumulated
        while z is not None:
            lats.append(z)
            z = self.enc_streaming.encode()
        self.num_encode_calls += 1
        assert lats, (f"TAEHV encoder emitted no latent: {x.shape[1]} frames in this call, "
                      f"{self.num_frames_in} so far (needs 4 per latent)")
        z = torch.cat(lats, 1)                                                  # [B, T_l, 16, h, w]
        if not is_scale and self.encode_space == "raw":
            z = z * self.std.to(z).view(1, 1, -1, 1, 1) + self.mean.to(z).view(1, 1, -1, 1, 1)
        return z.permute(0, 2, 1, 3, 4).contiguous()                            # [B, 16, T_l, h, w]

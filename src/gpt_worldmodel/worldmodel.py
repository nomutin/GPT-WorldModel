"""World Model Implementation."""

import math
from pathlib import Path

import torch
import wget
from beartype import beartype
from einops import rearrange
from einops.layers.torch import Rearrange
from jaxtyping import Float, Int, jaxtyped
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torch.nn import functional as tf
from torch.nn.functional import cross_entropy
from tqdm import tqdm

from gpt_worldmodel.factorization import (
    FactorizedEmbedding,
    cosine_schedule,
    factorize_token_ids,
)
from gpt_worldmodel.networks import Block


class GPTWorldModel(LightningModule):
    """
    GPT-style World Model with Vector Quantization.

    References
    ----------
    [1] https://arxiv.org/abs/2202.04200
    [2] https://arxiv.org/abs/2402.15391
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        num_heads: int,
        num_layers: int,
        embed_size: int,
        max_length: int,
        action_size: int,
        factored_vocab_size: int,
        num_factored_vocabs: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.max_length = max_length
        self.factored_vocab_size = factored_vocab_size
        self.num_factored_vocabs = num_factored_vocabs

        self.action_embedding = nn.Sequential(
            nn.Linear(action_size, embed_size),
            Rearrange("b l d -> b l 1 1 d"),
        )
        self.token_embedding = FactorizedEmbedding(
            factored_vocab_size=factored_vocab_size,
            num_factored_vocabs=num_factored_vocabs,
            d_model=embed_size,
            mask_token_id=factored_vocab_size**num_factored_vocabs,
        )
        self.dynamics = nn.ModuleList(
            [Block(d_model=embed_size, num_heads=num_heads) for _ in range(num_layers)],
        )
        self.output_proj = nn.Linear(embed_size, num_factored_vocabs * factored_vocab_size, bias=False)

    @jaxtyped(typechecker=beartype)
    def compute_logits(
        self,
        actions: Float[Tensor, "B L D"],
        codes: Int[Tensor, "B L H W"],
    ) -> Float[Tensor, "B L H W V"]:
        """
        Compute logits from actions and codes.

        Parameters
        ----------
        actions : Float[Tensor, "B L D"]
            Action sequence (t = 0~L-1).
        codes : Int[Tensor, "B L H W"]
            Codes sequence (t = 0~L-1).

        Returns
        -------
        Float[Tensor, "B L H W V"]
            Logit sequence (t = 1~L).
        """
        code_embeds = self.token_embedding.forward(codes)
        action_embeds = self.action_embedding(actions)
        logits = code_embeds + action_embeds
        for attn in self.dynamics:
            logits = attn(logits)
        return self.output_proj.forward(logits)

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def generate(
        self,
        context_actions: Float[Tensor, "B L1 D"],
        context_codes: Int[Tensor, "B L1 H W"],
        query_actions: Float[Tensor, "B L2 D"],
    ) -> Int[Tensor, "B L1+L2 H W"]:
        """
        Generate codes from context.

        Parameters
        ----------
        context_actions : Float[Tensor, "B L1 D"]
            Context action sequence.
        context_codes : Int[Tensor, "B L1 H W"]
            Context codes sequence.
        query_actions : Float[Tensor, "B L2 D"]
            Query action sequence.

        Returns
        -------
        Int[Tensor, "B L1+L2 H W"]
            Generated codes.
        """
        for t in tqdm(range(query_actions.shape[1])):
            next_codes = self.maskgit_generate(
                indices=context_codes[:, -self.max_length :],
                actions=context_actions[:, -self.max_length :],
                maskgit_steps=2,
            )
            next_codes = rearrange(next_codes, "b h w -> b 1 h w")
            context_codes = torch.cat([context_codes, next_codes], dim=1)
            next_action = query_actions[:, [t]]
            context_actions = torch.cat([context_actions, next_action], dim=1)
        return context_codes

    @torch.no_grad()
    def maskgit_generate(
        self,
        indices: Int[Tensor, "B L H W"],
        actions: Float[Tensor, "B L D"],
        maskgit_steps: int,
    ) -> Int[Tensor, "B L H W"]:
        """
        Perform MaskGIT-style inference.

        Parameters
        ----------
        indices : Int[Tensor, "B L H W"]
            Input tokens.
            Frames timestep and later must be masked
        actions : Float[Tensor, "B L D"]
            Actions.
        maskgit_steps : int, optional
            The number of MaskGIT-style inference steps to take.

        Returns
        -------
        Int[Tensor, "B L H W"]
            Sampled unfactorized tokens.
        """
        batch_size, _, height, width = indices.shape

        unmasked = torch.zeros(batch_size, height * width, dtype=torch.bool, device=self.device)
        samples = torch.zeros((batch_size, height, width), dtype=torch.long, device=self.device)

        for step in range(maskgit_steps):
            factored_logits = self.compute_logits(codes=indices, actions=actions)[:, -1]
            factored_logits = rearrange(
                tensor=factored_logits,
                pattern="b h w (num_vocabs vocab_size) -> b vocab_size num_vocabs h w",
                vocab_size=self.factored_vocab_size,
                num_vocabs=self.num_factored_vocabs,
            )
            samples = torch.zeros((batch_size, height, width), dtype=torch.int, device=self.device)
            confidences = torch.ones((batch_size, height, width), dtype=torch.float, device=self.device)

            for probs in tf.softmax(factored_logits, dim=1).flip(2).unbind(2):
                sample = probs.argmax(dim=1)
                samples *= self.factored_vocab_size
                samples += sample
                confidences *= torch.gather(probs, 1, sample.unsqueeze(1)).squeeze(1)

            prev_unmasked = unmasked.clone()
            samples_flat = rearrange(samples, "b h w -> b (h w)")

            if step != maskgit_steps - 1:
                confidences_flat = rearrange(confidences, "b h w -> b (h w)")
                confidences_flat[unmasked] = torch.inf
                least_confident_tokens = torch.argsort(confidences_flat, dim=1)
                n = math.ceil(cosine_schedule((step + 1) / maskgit_steps) * height * width)
                unmasked.scatter_(1, least_confident_tokens[:, n:], value=True)
                samples_flat.scatter_(
                    dim=1,
                    index=least_confident_tokens[:, :n],
                    value=self.factored_vocab_size**self.num_factored_vocabs,
                )

            prev_img_flat = rearrange(indices[:, -1], "b h w -> b (h w)")
            samples_flat[prev_unmasked] = prev_img_flat[prev_unmasked]
            samples = rearrange(samples_flat, "b (h w) -> b h w", h=height, w=width)

            indices[:, -1] = samples

        return samples

    def step(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """
        Shared training/validation step.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            actions, codes. Shape: (B L D), (B L H W)

        Returns
        -------
        dict[str, Tensor]
            Losses.

        """
        action_inputs, _, codes_input, codes_target = batch

        logits = self.compute_logits(actions=action_inputs, codes=codes_input)
        logits = rearrange(
            logits,
            "b l h w (num_vocabs vocab_size) -> b vocab_size l h w num_vocabs",
            num_vocabs=self.num_factored_vocabs,
            vocab_size=self.factored_vocab_size,
        )
        codes_target = factorize_token_ids(
            token_ids=codes_target,
            num_factored_vocabs=self.num_factored_vocabs,
            factored_vocab_size=self.factored_vocab_size,
        )
        loss = cross_entropy(input=logits, target=codes_target)
        return {"loss": loss}

    def training_step(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """
        Run training step.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            Batch tensors.

        Returns
        -------
        dict[str, Tensor]
            Losses.

        """
        loss_dict = self.step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def validation_step(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """
        Run validation step.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            Batch tensors.

        Returns
        -------
        dict[str, Tensor]
            Losses. Keys are prefixed with "val_".

        """
        loss_dict = self.step(batch)
        loss_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict


class LogWorldModelGenerations(Callback):
    """Callback to log `GPTWorldModel` outputs to WandbLogger."""

    def __init__(self, every_n_epochs: int, num_samples: int, fps: int, decoder_name: str) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.fps = fps

        local_dir = Path(".model_cache") / decoder_name
        decoder_path = local_dir / "decoder.jit"
        if not decoder_path.exists():
            local_dir.mkdir(exist_ok=True, parents=True)
            url = f"https://huggingface.co/nvidia/{decoder_name}/resolve/main/decoder.jit?download=true"
            wget.download(url, str(decoder_path))
        self.decoder = torch.jit.load(decoder_path).eval()  # type: ignore[no-untyped-call]
        self.decoder.eval()

    def _get_predict_batch(self, trainer: Trainer, pl_module: LightningModule) -> tuple[Tensor, ...]:
        dataloader = trainer.datamodule.predict_dataloader()  # type: ignore[attr-defined]
        return tuple(b[: self.num_samples].to(pl_module.device) for b in next(iter(dataloader)))

    @staticmethod
    def _process_observation(observation: Int[Tensor, "B C L H W"]) -> list[Int[Tensor, "L C H W"]]:
        """
        Process batched observation into WandbLogger.log_video format.

        Parameters
        ----------
        observation : Int[Tensor, "B C L H W"]
            Batched observation.

        Returns
        -------
        list[Int[Tensor, "L C H W"]]
            List of processed observations.
        """
        observation = rearrange(observation, "b c l h w -> b l c h w")
        return list(observation.clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu())

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Log validation results to WandbLogger.

        Parameters
        ----------
        trainer : Trainer
            Trainer which has WandbLogger and predict_dataloader().
        pl_module : LightningModule
            LightningModule(GPTWorldModel).

        Raises
        ------
        TypeError
            If the logger is not WandbLogger.
        """
        if not isinstance(logger := trainer.logger, WandbLogger):
            msg = "LogWorldModelGenerations requires WandbLogger."
            raise TypeError(msg)
        if trainer.current_epoch % self.every_n_epochs != 0 or trainer.current_epoch <= 1:
            return

        self.decoder.to(pl_module.device)
        action_inputs, _, codes_input, codes_target = self._get_predict_batch(trainer, pl_module=pl_module)

        target = self.decoder(codes_target)
        target_list = self._process_observation(target)

        generated_indices = pl_module.generate(
            context_actions=action_inputs[:, :pl_module.max_length],
            context_codes=codes_input[:, :pl_module.max_length],
            query_actions=action_inputs[:, pl_module.max_length:],
        )
        generated = self.decoder(generated_indices)
        generated_list = self._process_observation(generated)

        logger.log_video("target", videos=target_list, fps=[self.fps] * self.num_samples)
        logger.log_video("recon", videos=generated_list, fps=[self.fps] * self.num_samples)

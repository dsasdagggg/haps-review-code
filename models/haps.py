from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_bool_flag(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def destationary_norm(
    x: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=1, keepdim=True).detach()
    x0 = x - mean
    var = torch.var(x0, dim=1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps).detach()
    return x0 / std, mean, std


def destationary_denorm(y_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return y_norm * std + mean


def horizon_diff(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return x[:, 1:, :] - x[:, :-1, :]


class FourierCalendarEncoder(nn.Module):
    def __init__(self, d_model: int, n_freqs: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_freqs = int(n_freqs)
        in_dim = 4 + 4 * 2 * self.n_freqs
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def _ensure_mark4(self, x_mark: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_mark is None:
            return None
        squeeze_back = False
        if x_mark.dim() == 2:
            x_mark = x_mark.unsqueeze(1)
            squeeze_back = True

        B, L, M = x_mark.shape
        if M == 4:
            out = x_mark
        elif M > 4:
            out = x_mark[:, :, :4].contiguous()
        else:
            pad = torch.zeros(B, L, 4 - M, device=x_mark.device, dtype=x_mark.dtype)
            out = torch.cat([x_mark, pad], dim=-1)

        if squeeze_back:
            out = out.squeeze(1)
        return out

    def forward(self, x_mark: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_mark is None:
            return None

        squeeze_back = False
        if x_mark.dim() == 2:
            x_mark = x_mark.unsqueeze(1)
            squeeze_back = True

        x_mark = self._ensure_mark4(x_mark).float()
        scales = x_mark.new_tensor([12.0, 31.0, 7.0, 24.0]).view(1, 1, 4)
        z = x_mark / scales

        feats = [x_mark]
        for k in range(1, self.n_freqs + 1):
            ang = 2.0 * math.pi * k * z
            feats.append(torch.sin(ang))
            feats.append(torch.cos(ang))

        h = torch.cat(feats, dim=-1)
        out = self.norm(self.proj(h))

        if squeeze_back:
            out = out.squeeze(1)
        return out


class HistoryTokenizer(nn.Module):
    def __init__(self, seq_len: int, c_in: int, d_model: int, dropout: float):
        super().__init__()
        self.seq_len = int(seq_len)
        self.c_in = int(c_in)
        self.d_model = int(d_model)

        self.hist_proj = nn.Linear(seq_len, d_model)
        self.var_embed = nn.Parameter(torch.randn(1, c_in, d_model) * 0.02)

        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                      
        xc = x.transpose(1, 2).contiguous()             
        tok = self.hist_proj(xc)                        
        tok = tok + self.var_embed[:, :tok.shape[1], :].to(dtype=tok.dtype)
        return self.out(tok)


class HAPSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        pred_mean: torch.Tensor,
        innov_mean: torch.Tensor,
        rank_gates: torch.Tensor,
        band_decay: torch.Tensor,
        mono_penalty: torch.Tensor,
        weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        l_sup = F.mse_loss(y_hat, y_true)
        l_pred = F.mse_loss(pred_mean, y_true)

                                                 
                                                      

        if y_true.shape[1] > 1:
            l_dyn = F.smooth_l1_loss(horizon_diff(y_hat), horizon_diff(y_true))
        else:
            l_dyn = torch.zeros_like(l_sup)

        tau_rank = torch.linspace(
            0.0, 1.0, rank_gates.shape[0], device=rank_gates.device, dtype=rank_gates.dtype
        ).view(-1, 1)
        l_rank = (tau_rank * rank_gates).mean()

        tau_band = torch.linspace(
            0.0, 1.0, band_decay.shape[0], device=band_decay.device, dtype=band_decay.dtype
        ).view(-1, 1)
        l_band = (tau_band * band_decay).mean()

        l_innov = innov_mean.pow(2).mean()
        l_gate = mono_penalty

        l_total = (
            l_sup
            + weights["lambda_pred"] * l_pred
            + weights["lambda_dyn"] * l_dyn
            + weights["lambda_rank"] * (l_rank + 0.5 * l_band)
            + weights["lambda_gate"] * l_gate
            + weights["lambda_innov"] * l_innov
        )

        return {
            "L_total": l_total,
            "L_sup": l_sup.detach(),
            "L_pred": l_pred.detach(),
            "L_dyn": l_dyn.detach(),
            "L_rank": l_rank.detach(),
            "L_band": l_band.detach(),
            "L_gate": l_gate.detach(),
            "L_innov": l_innov.detach(),
        }



class HorizonAdaptivePredictableSubspaceCore(nn.Module):
    VALID_VARIANTS = {
        "full",
        "no_subspace",
        "static_spectral",
        "no_spectral",
        "no_innovation",
        "fixed_rank",
        "generic_decoder",
    }

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        c_in: int,
        c_out: int,
        d_model: int,
        d_ff: int,
        rank: int,
        n_bands: int,
        dropout: float,
        variant: str = "full",
    ):
        super().__init__()

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.c_in = int(c_in)
        self.c_out = int(c_out)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.rank = int(rank)
        self.n_bands = int(n_bands)
        self.dropout = float(dropout)

        self.variant = str(variant).strip().lower()

        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown HAPS variant: {self.variant}. "
                f"Valid choices: {sorted(self.VALID_VARIANTS)}"
            )

                                                              
                        
                                                              

        self.ablate_subspace = (
            self.variant == "no_subspace"
        )

        self.static_spectral = (
            self.variant == "static_spectral"
        )

        self.ablate_spectral_branch = (
            self.variant == "no_spectral"
        )

        self.ablate_innovation = (
            self.variant == "no_innovation"
        )

        self.fixed_rank = (
            self.variant == "fixed_rank"
        )

        self.generic_decoder_mode = (
            self.variant == "generic_decoder"
        )

                                                              
                        
         
                                                           
                                                  
                                                              

        self.tokenizer = HistoryTokenizer(
            seq_len=seq_len,
            c_in=c_in,
            d_model=d_model,
            dropout=dropout,
        )

        self.global_refine = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )

        self.horizon_embed = nn.Parameter(
            torch.randn(
                1,
                pred_len,
                d_model,
            ) * 0.02
        )

        self.calendar_enc = FourierCalendarEncoder(
            d_model=d_model,
            n_freqs=4,
            dropout=dropout,
        )

        self.channel_proj = (
            nn.Identity()
            if c_out == c_in
            else nn.Linear(c_in, c_out)
        )

                                                              
                        
                                                              

        self.generic_hidden_dim = 0
        self.generic_target_head_params = 0
        self.generic_actual_head_params = 0
        self.generic_parameter_gap = 0.0

        if self.generic_decoder_mode:
            target_params = (
                self._structured_head_parameter_count()
            )

            hidden_dim = (
                self._select_generic_hidden_dim(
                    target_params
                )
            )

            self.generic_var_decoder = (
                self._make_generic_decoder(
                    hidden_dim
                )
            )

            self.generic_global_decoder = (
                self._make_generic_decoder(
                    hidden_dim
                )
            )

            self.generic_hidden_dim = int(
                hidden_dim
            )

            self.generic_target_head_params = int(
                target_params
            )

            self.generic_actual_head_params = int(
                self._count_parameters(
                    self.generic_var_decoder
                )
                +
                self._count_parameters(
                    self.generic_global_decoder
                )
            )

            self.generic_parameter_gap = float(
                abs(
                    self.generic_actual_head_params
                    -
                    self.generic_target_head_params
                )
                /
                max(
                    1,
                    self.generic_target_head_params,
                )
            )

        else:
            self._build_structured_heads()

                                                              
                              
                                                              

        freq_bins = seq_len // 2 + 1

        bounds = self._build_freq_bounds(
            freq_bins=freq_bins,
            n_bands=n_bands,
        )

        masks = torch.zeros(
            n_bands,
            freq_bins,
        )

        for k in range(n_bands):
            start = int(bounds[k].item())
            end = int(bounds[k + 1].item())
            masks[k, start:end] = 1.0

        self.register_buffer(
            "fft_band_masks",
            masks,
            persistent=False,
        )

                                                              
                                         
                                                              

    @staticmethod
    def _count_parameters(
        module: nn.Module,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter in module.parameters()
        )

    @staticmethod
    def _linear_parameter_count(
        in_features: int,
        out_features: int,
    ) -> int:
        return int(
            in_features * out_features
            +
            out_features
        )

    def _structured_head_parameter_count(
        self,
    ) -> int:
        """
        Count the structured HAPS parameters replaced by A6.

        The following shared modules are excluded:
          tokenizer
          global_refine
          horizon_embed
          calendar_enc
          channel_proj
        """

        d_model = self.d_model
        rank = self.rank
        n_bands = self.n_bands
        seq_len = self.seq_len
        pred_len = self.pred_len

        layer_norm = 2 * d_model

        coefficient_head = (
            layer_norm
            +
            self._linear_parameter_count(
                d_model,
                d_model,
            )
            +
            self._linear_parameter_count(
                d_model,
                rank,
            )
        )

        coefficient_heads = (
            2 * coefficient_head
        )

        rank_program = (
            pred_len * rank
            +
            2 * rank
        )

        one_band_gate_head = (
            layer_norm
            +
            self._linear_parameter_count(
                d_model,
                n_bands,
            )
        )

        band_gate_heads = (
            2 * one_band_gate_head
        )

        band_gate_parameters = (
            3 * n_bands
        )

        band_predictors = (
            n_bands
            *
            self._linear_parameter_count(
                seq_len,
                pred_len,
            )
        )

        innovation = (
            self._linear_parameter_count(
                seq_len,
                pred_len,
            )
            +
            3
        )

        program_to_rank = (
            layer_norm
            +
            self._linear_parameter_count(
                d_model,
                rank,
            )
        )

        return int(
            coefficient_heads
            +
            rank_program
            +
            band_gate_heads
            +
            band_gate_parameters
            +
            band_predictors
            +
            innovation
            +
            program_to_rank
        )

    def _generic_decoder_parameter_count(
        self,
        hidden_dim: int,
    ) -> int:
        d_model = self.d_model
        pred_len = self.pred_len
        hidden_dim = int(hidden_dim)

        one_decoder = (
            2 * d_model
            +
            self._linear_parameter_count(
                d_model,
                hidden_dim,
            )
            +
            self._linear_parameter_count(
                hidden_dim,
                pred_len,
            )
        )

        return int(
            2 * one_decoder
        )

    def _select_generic_hidden_dim(
        self,
        target_params: int,
    ) -> int:
        """
        Search a nearby hidden width that minimizes parameter
        mismatch between the structured HAPS heads and A6.
        """

        denominator = max(
            1,
            2 * (
                self.d_model
                +
                self.pred_len
                +
                1
            ),
        )

        base = 2 * (
            2 * self.d_model
            +
            self.pred_len
        )

        estimate = max(
            1,
            int(
                round(
                    (
                        target_params
                        -
                        base
                    )
                    /
                    denominator
                )
            ),
        )

        lower = max(
            1,
            estimate - 512,
        )

        upper = max(
            lower + 1,
            estimate + 513,
        )

        candidates = range(
            lower,
            upper,
        )

        return min(
            candidates,
            key=lambda width: abs(
                self._generic_decoder_parameter_count(
                    width
                )
                -
                target_params
            ),
        )

    def _make_generic_decoder(
        self,
        hidden_dim: int,
    ) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(
                self.d_model
            ),
            nn.Linear(
                self.d_model,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                self.dropout
            ),
            nn.Linear(
                hidden_dim,
                self.pred_len,
            ),
        )

                                                              
                                
                                                              

    def _build_structured_heads(
        self,
    ) -> None:
        d_model = self.d_model
        rank = self.rank
        n_bands = self.n_bands

        self.coeff_var_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(
                self.dropout
            ),
            nn.Linear(
                d_model,
                rank,
            ),
        )

        self.coeff_global_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(
                self.dropout
            ),
            nn.Linear(
                d_model,
                rank,
            ),
        )

        self.rank_basis = nn.Parameter(
            torch.randn(
                self.pred_len,
                rank,
            ) * 0.02
        )

        self.rank_start = nn.Parameter(
            torch.linspace(
                3.0,
                -1.0,
                rank,
            )
        )

        self.rank_decay = nn.Parameter(
            torch.ones(rank)
        )

        self.var_band_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(
                d_model,
                n_bands,
            ),
        )

        self.horizon_band_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(
                d_model,
                n_bands,
            ),
        )

        self.band_strength_scale = nn.Parameter(
            torch.ones(n_bands)
        )

        self.band_start = nn.Parameter(
            torch.linspace(
                3.0,
                0.0,
                n_bands,
            )
        )

        self.band_decay = nn.Parameter(
            torch.linspace(
                0.5,
                2.0,
                n_bands,
            )
        )

        self.band_heads = nn.ModuleList(
            [
                nn.Linear(
                    self.seq_len,
                    self.pred_len,
                )
                for _ in range(n_bands)
            ]
        )

        self.innovation_head = nn.Linear(
            self.seq_len,
            self.pred_len,
        )

        self.innovation_scale = nn.Parameter(
            torch.tensor(0.05)
        )

        self.innov_start = nn.Parameter(
            torch.tensor(-2.0)
        )

        self.innov_slope = nn.Parameter(
            torch.tensor(1.0)
        )

        self.program_to_rank = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(
                d_model,
                rank,
            ),
            nn.Tanh(),
        )

                                                              
                     
                                                              

    @staticmethod
    def _build_freq_bounds(
        freq_bins: int,
        n_bands: int,
    ) -> torch.Tensor:
        grid = torch.linspace(
            0.0,
            1.0,
            n_bands + 1,
        )

        bounds = torch.round(
            (grid ** 2)
            *
            freq_bins
        ).long()

        bounds[0] = 0
        bounds[-1] = freq_bins

        for index in range(
            1,
            len(bounds),
        ):
            if (
                bounds[index]
                <=
                bounds[index - 1]
            ):
                bounds[index] = min(
                    freq_bins,
                    bounds[index - 1] + 1,
                )

        bounds[-1] = freq_bins
        return bounds

                                                              
                
                                                              

    def _rank_gates(
        self,
        device,
        dtype,
    ) -> torch.Tensor:
                                                              
                             
         
                                                            
                                                
                                                              

        if self.fixed_rank:
            static_gate = torch.sigmoid(
                self.rank_start
            ).to(
                device=device,
                dtype=dtype,
            )

            return static_gate.view(
                1,
                self.rank,
            ).expand(
                self.pred_len,
                -1,
            )

        tau = torch.linspace(
            0.0,
            1.0,
            self.pred_len,
            device=device,
            dtype=dtype,
        ).view(
            self.pred_len,
            1,
        )

        decay = F.softplus(
            self.rank_decay
        ).view(
            1,
            self.rank,
        )

        start = self.rank_start.view(
            1,
            self.rank,
        )

        return torch.sigmoid(
            start
            -
            5.0 * tau * decay
        )

    def _band_decay_gates(
        self,
        device,
        dtype,
    ) -> torch.Tensor:
        tau = torch.linspace(
            0.0,
            1.0,
            self.pred_len,
            device=device,
            dtype=dtype,
        ).view(
            self.pred_len,
            1,
        )

        decay = F.softplus(
            self.band_decay
        ).view(
            1,
            self.n_bands,
        )

        start = self.band_start.view(
            1,
            self.n_bands,
        )

        return torch.sigmoid(
            start
            -
            5.0 * tau * decay
        )

    def _innovation_horizon_gate(
        self,
        device,
        dtype,
    ) -> torch.Tensor:
        tau = torch.linspace(
            0.0,
            1.0,
            self.pred_len,
            device=device,
            dtype=dtype,
        )

        gate = torch.sigmoid(
            self.innov_start
            +
            5.0
            *
            F.softplus(
                self.innov_slope
            )
            *
            tau
        )

        return gate.view(
            1,
            self.pred_len,
            1,
        )

                                                              
                                     
                                                              

    def _spectral_decompose(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        FFT is evaluated in FP32.

        This avoids half-precision FFT failures on older GPUs,
        including NVIDIA P100.
        """

        original_dtype = x.dtype

        x_fft = torch.fft.rfft(
            x.float(),
            dim=1,
            norm="ortho",
        )

        outputs = []

        for band_index in range(
            self.n_bands
        ):
            mask = self.fft_band_masks[
                band_index
            ].to(
                x_fft.real.dtype
            ).view(
                1,
                -1,
                1,
            )

            band = torch.fft.irfft(
                x_fft * mask,
                n=self.seq_len,
                dim=1,
                norm="ortho",
            )

            outputs.append(
                band.to(
                    dtype=original_dtype
                )
            )

        return outputs

                                                              
                                     
                                                              

    def _horizon_program(
        self,
        batch_size: int,
        x_mark_future: Optional[torch.Tensor],
        device,
        dtype,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        h_embed = self.horizon_embed[
            :,
            :self.pred_len,
            :,
        ].expand(
            batch_size,
            -1,
            -1,
        ).to(
            device=device,
            dtype=dtype,
        )

        if x_mark_future is None:
            return h_embed, None

        calendar = self.calendar_enc(
            x_mark_future
        ).to(
            device=device,
            dtype=dtype,
        )

        h_embed = (
            h_embed
            +
            calendar
        )

        return h_embed, calendar

                                                              
                        
                                                              

    def _forward_generic(
        self,
        tokens: torch.Tensor,
        global_state: torch.Tensor,
        h_embed: torch.Tensor,
        device,
        dtype,
    ) -> Dict[str, torch.Tensor]:
        """
        A6 capacity-matched non-structured decoder.

        Retains:
          tokenizer
          global representation
          horizon embedding
          calendar representation

        Replaces:
          low-rank basis
          rank contraction
          spectral routing
          innovation branch
        """

        variable_forecast = (
            self.generic_var_decoder(
                tokens
            )
            .transpose(
                1,
                2,
            )
        )

        global_forecast = (
            self.generic_global_decoder(
                global_state
            )
            .unsqueeze(-1)
        )

                                                         
        horizon_context = (
            0.05
            *
            h_embed.mean(
                dim=-1,
                keepdim=True,
            )
        )

        pred_mean_in = (
            variable_forecast
            +
            global_forecast
            +
            horizon_context
        )

        innov_mean_in = torch.zeros_like(
            pred_mean_in
        )

        y_hat_in = pred_mean_in

        rank_gates = torch.ones(
            self.pred_len,
            self.rank,
            device=device,
            dtype=dtype,
        )

        band_decay = torch.ones(
            self.pred_len,
            self.n_bands,
            device=device,
            dtype=dtype,
        )

        zero = torch.zeros(
            (),
            device=device,
            dtype=dtype,
        )

        return {
            "y_hat": self.channel_proj(
                y_hat_in
            ),
            "pred_mean": self.channel_proj(
                pred_mean_in
            ),
            "innov_mean": self.channel_proj(
                innov_mean_in
            ),
            "stats": {
                "rank_gates": rank_gates,
                "band_decay": band_decay,
                "mono_penalty": zero,
                "rank_first": (
                    rank_gates[0]
                    .sum()
                    .detach()
                ),
                "rank_last": (
                    rank_gates[-1]
                    .sum()
                    .detach()
                ),
                "band_gate_mean": zero,
            },
        }

                                                              
             
                                                              

    def forward(
        self,
        x_norm: torch.Tensor,
        x_mark_future: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, channels = (
            x_norm.shape
        )

        dtype = x_norm.dtype
        device = x_norm.device

        tokens = self.tokenizer(
            x_norm
        )

        global_state = self.global_refine(
            tokens.mean(
                dim=1
            )
        )

        h_embed, _ = self._horizon_program(
            batch_size=batch_size,
            x_mark_future=x_mark_future,
            device=device,
            dtype=dtype,
        )

                                                              
            
                                                              

        if self.generic_decoder_mode:
            return self._forward_generic(
                tokens=tokens,
                global_state=global_state,
                h_embed=h_embed,
                device=device,
                dtype=dtype,
            )

                                                              
                                     
                                                              

        coeff_var = self.coeff_var_head(
            tokens
        )

        coeff_global = (
            self.coeff_global_head(
                global_state
            )
            .unsqueeze(1)
        )

        coeff = (
            coeff_var
            +
            coeff_global
        )

        rank_gates = self._rank_gates(
            device=device,
            dtype=dtype,
        )

        basis = (
            self.rank_basis.to(
                dtype=dtype
            )
            *
            rank_gates
        )

        subspace_hat = torch.einsum(
            "bcr,hr->bhc",
            coeff,
            basis,
        )

        prog_bias = self.program_to_rank(
            h_embed
        )

        prog_term = torch.einsum(
            "bhr,bcr->bhc",
            prog_bias,
            coeff,
        )

        subspace_hat = (
            subspace_hat
            +
            0.05 * prog_term
        )

                                                              
            
                                                              

        if self.ablate_subspace:
            subspace_hat = torch.zeros_like(
                subspace_hat
            )

                                                              
                         
                                                              

        band_decay = self._band_decay_gates(
            device=device,
            dtype=dtype,
        )

        spectral_hat = torch.zeros(
            batch_size,
            self.pred_len,
            channels,
            device=device,
            dtype=dtype,
        )

        mono_terms = []
        band_means = []

                                                              
                                                     
         
                                                             
                                                            
                                                              

        if not self.ablate_spectral_branch:
            bands = self._spectral_decompose(
                x_norm
            )

            var_band_logits = (
                self.var_band_head(
                    tokens
                )
            )

            h_band_logits = (
                self.horizon_band_head(
                    h_embed
                )
            )

            for band_index in range(
                self.n_bands
            ):
                band_input = bands[
                    band_index
                ]

                band_prediction = (
                    self.band_heads[
                        band_index
                    ](
                        band_input.transpose(
                            1,
                            2,
                        )
                    )
                    .transpose(
                        1,
                        2,
                    )
                )

                                                              
                                            
                 
                                                          
                                      
                           
                             
                                     
                 
                                              
                                                              

                if self.static_spectral:
                    static_gate = torch.sigmoid(
                        self.band_start[
                            band_index
                        ]
                    ).to(
                        dtype=dtype
                    )

                    gamma = static_gate.view(
                        1,
                        1,
                        1,
                    ).expand(
                        batch_size,
                        self.pred_len,
                        channels,
                    )

                else:
                    strength = torch.log1p(
                        band_input
                        .abs()
                        .mean(
                            dim=1
                        )
                    )

                    logits = (
                        var_band_logits[
                            :,
                            :,
                            band_index,
                        ].unsqueeze(1)
                        +
                        h_band_logits[
                            :,
                            :,
                            band_index,
                        ].unsqueeze(2)
                        +
                        self.band_strength_scale[
                            band_index
                        ]
                        *
                        strength.unsqueeze(1)
                    )

                    gamma = (
                        torch.sigmoid(
                            logits
                        )
                        *
                        band_decay[
                            :,
                            band_index,
                        ].view(
                            1,
                            self.pred_len,
                            1,
                        )
                    )

                spectral_hat = (
                    spectral_hat
                    +
                    gamma
                    *
                    band_prediction
                )

                band_means.append(
                    gamma.mean()
                )

                if self.pred_len > 1:
                    mono_terms.append(
                        F.relu(
                            gamma[
                                :,
                                1:,
                                :,
                            ]
                            -
                            gamma[
                                :,
                                :-1,
                                :,
                            ]
                        ).mean()
                    )

        else:
            band_decay = torch.zeros_like(
                band_decay
            )

        mono_penalty = (
            torch.stack(
                mono_terms
            ).mean()
            if mono_terms
            else
            torch.zeros(
                (),
                device=device,
                dtype=dtype,
            )
        )

        pred_mean_in = (
            subspace_hat
            +
            spectral_hat
        )

                                                              
                           
                                                              

        innovation_gate = (
            self._innovation_horizon_gate(
                device=device,
                dtype=dtype,
            )
        )

        innov_mean_in = (
            self.innovation_scale
            *
            innovation_gate
            *
            self.innovation_head(
                x_norm.transpose(
                    1,
                    2,
                )
            ).transpose(
                1,
                2,
            )
        )

                                                              
            
                                                              

        if self.ablate_innovation:
            innov_mean_in = torch.zeros_like(
                innov_mean_in
            )

        y_hat_in = (
            pred_mean_in
            +
            innov_mean_in
        )

        zero = torch.zeros(
            (),
            device=device,
            dtype=dtype,
        )

        stats = {
            "rank_gates": rank_gates,
            "band_decay": band_decay,
            "mono_penalty": mono_penalty,
            "rank_first": (
                rank_gates[0]
                .sum()
                .detach()
            ),
            "rank_last": (
                rank_gates[-1]
                .sum()
                .detach()
            ),
            "band_gate_mean": (
                torch.stack(
                    band_means
                )
                .mean()
                .detach()
                if band_means
                else zero
            ),
        }

        return {
            "y_hat": self.channel_proj(
                y_hat_in
            ),
            "pred_mean": self.channel_proj(
                pred_mean_in
            ),
            "innov_mean": self.channel_proj(
                innov_mean_in
            ),
            "stats": stats,
        }


class ForecastPredictableSubspaceModel(nn.Module):
    VALID_VARIANTS = (
        HorizonAdaptivePredictableSubspaceCore
        .VALID_VARIANTS
    )

    LEGACY_FLAG_TO_VARIANT = {
        "haps_ablate_subspace":
            "no_subspace",

        "haps_ablate_spectral_gate":
            "static_spectral",

        "haps_static_spectral_routing":
            "static_spectral",

        "haps_ablate_spectral_branch":
            "no_spectral",

        "haps_ablate_innovation":
            "no_innovation",

        "haps_fixed_rank_gate":
            "fixed_rank",

        "haps_capacity_matched_generic":
            "generic_decoder",
    }

    @classmethod
    def _resolve_variant(
        cls,
        args,
    ) -> str:
        explicit = str(
            getattr(
                args,
                "haps_variant",
                "full",
            )
        ).strip().lower()

        if explicit not in cls.VALID_VARIANTS:
            raise ValueError(
                f"Unknown --haps_variant={explicit}. "
                f"Valid choices: "
                f"{sorted(cls.VALID_VARIANTS)}"
            )

        active_legacy = [
            variant
            for flag, variant
            in cls.LEGACY_FLAG_TO_VARIANT.items()
            if _as_bool_flag(
                getattr(
                    args,
                    flag,
                    0,
                )
            )
        ]

        active_legacy = sorted(
            set(
                active_legacy
            )
        )

        if len(active_legacy) > 1:
            raise ValueError(
                "HAPS variants are mutually exclusive. "
                "Active legacy variants: "
                f"{active_legacy}. "
                "Use one --haps_variant."
            )

        if active_legacy:
            legacy_variant = (
                active_legacy[0]
            )

            if (
                explicit != "full"
                and
                explicit != legacy_variant
            ):
                raise ValueError(
                    "Conflicting HAPS variants: "
                    f"--haps_variant={explicit}, "
                    f"legacy flags request "
                    f"{legacy_variant}."
                )

            return legacy_variant

        return explicit

    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.seq_len = int(
            args.seq_len
        )

        self.label_len = int(
            args.label_len
        )

        self.pred_len = int(
            args.pred_len
        )

        self.enc_in = int(
            args.enc_in
        )

        self.c_out = int(
            getattr(
                args,
                "c_out",
                args.enc_in,
            )
        )

        self.d_model = int(
            getattr(
                args,
                "d_model",
                192,
            )
        )

        self.d_ff = int(
            getattr(
                args,
                "d_ff",
                768,
            )
        )

        self.dropout = float(
            getattr(
                args,
                "dropout",
                0.2,
            )
        )

        self.rank = int(
            getattr(
                args,
                "haps_rank",
                16,
            )
        )

        self.n_bands = int(
            getattr(
                args,
                "haps_n_bands",
                4,
            )
        )

        self.lambda_pred = float(
            getattr(
                args,
                "haps_lambda_pred",
                0.20,
            )
        )

        self.lambda_dyn = float(
            getattr(
                args,
                "haps_lambda_dyn",
                0.10,
            )
        )

        self.lambda_rank = float(
            getattr(
                args,
                "haps_lambda_rank",
                1e-3,
            )
        )

        self.lambda_gate = float(
            getattr(
                args,
                "haps_lambda_gate",
                1e-3,
            )
        )

        self.lambda_innov = float(
            getattr(
                args,
                "haps_lambda_innov",
                1e-4,
            )
        )

        self.variant = self._resolve_variant(
            args
        )

                                                    
                   
                             
                       
                                        
        setattr(
            args,
            "haps_variant",
            self.variant,
        )

        self.core = (
            HorizonAdaptivePredictableSubspaceCore(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                c_in=self.enc_in,
                c_out=self.c_out,
                d_model=self.d_model,
                d_ff=self.d_ff,
                rank=self.rank,
                n_bands=self.n_bands,
                dropout=self.dropout,
                variant=self.variant,
            )
        )

        self.loss_pack = HAPSLoss()

    def _weights(
        self,
    ) -> Dict[str, float]:
        weights = {
            "lambda_pred":
                self.lambda_pred,

            "lambda_dyn":
                self.lambda_dyn,

            "lambda_rank":
                self.lambda_rank,

            "lambda_gate":
                self.lambda_gate,

            "lambda_innov":
                self.lambda_innov,
        }

        if self.variant == "no_innovation":
            weights[
                "lambda_innov"
            ] = 0.0

        if self.variant == "no_spectral":
            weights[
                "lambda_gate"
            ] = 0.0

        if self.variant == "generic_decoder":
                                                   
                                
             
                                                         
            weights[
                "lambda_rank"
            ] = 0.0

            weights[
                "lambda_gate"
            ] = 0.0

            weights[
                "lambda_innov"
            ] = 0.0

        return weights

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        y_true=None,
        epoch_idx_1based=None,
    ):
        x_mark_future = None

        if x_mark_dec is not None:
            x_mark_future = x_mark_dec[
                :,
                -self.pred_len:,
                :,
            ]

        x_norm, mean, std = (
            destationary_norm(
                x_enc
            )
        )

        core_output = self.core(
            x_norm=x_norm,
            x_mark_future=x_mark_future,
        )

        y_hat_norm = core_output[
            "y_hat"
        ]

        pred_mean_norm = core_output[
            "pred_mean"
        ]

        innov_mean_norm = core_output[
            "innov_mean"
        ]

        stats = core_output[
            "stats"
        ]

        if self.c_out == self.enc_in:
            y_hat = destationary_denorm(
                y_hat_norm,
                mean,
                std,
            )

            pred_mean = destationary_denorm(
                pred_mean_norm,
                mean,
                std,
            )

            innov_mean = (
                innov_mean_norm
                *
                std
            )

        else:
            y_hat = y_hat_norm
            pred_mean = pred_mean_norm
            innov_mean = innov_mean_norm

        batch_size = x_enc.shape[0]

        output = torch.zeros(
            batch_size,
            self.label_len
            +
            self.pred_len,
            self.c_out,
            device=x_enc.device,
            dtype=y_hat.dtype,
        )

        output[
            :,
            -self.pred_len:,
            :,
        ] = y_hat

        if y_true is None:
            return output

        if (
            y_true.dim() == 3
            and
            y_true.shape[1]
            ==
            self.label_len
            +
            self.pred_len
        ):
            y_future = y_true[
                :,
                -self.pred_len:,
                :,
            ].contiguous()

        elif (
            y_true.dim() == 3
            and
            y_true.shape[1]
            ==
            self.pred_len
        ):
            y_future = y_true

        else:
            raise ValueError(
                "Unexpected y_true shape: "
                f"{tuple(y_true.shape)}"
            )

        loss_dict = self.loss_pack(
            y_hat=y_hat,
            y_true=y_future,
            pred_mean=pred_mean,
            innov_mean=innov_mean,
            rank_gates=stats[
                "rank_gates"
            ],
            band_decay=stats[
                "band_decay"
            ],
            mono_penalty=stats[
                "mono_penalty"
            ],
            weights=self._weights(),
        )

        loss_dict[
            "rank_first"
        ] = stats[
            "rank_first"
        ]

        loss_dict[
            "rank_last"
        ] = stats[
            "rank_last"
        ]

        loss_dict[
            "band_gate_mean"
        ] = stats[
            "band_gate_mean"
        ]

        loss_dict[
            "variant_id"
        ] = torch.tensor(
            float(
                sorted(
                    self.VALID_VARIANTS
                ).index(
                    self.variant
                )
            ),
            device=x_enc.device,
        )

        return output, loss_dict


Model = ForecastPredictableSubspaceModel
haps = ForecastPredictableSubspaceModel


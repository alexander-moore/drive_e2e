"""
TransformerPlanner — encoder-decoder trajectory planner.

Treats the 41-step kinematic history as a sequence of 7-dimensional tokens,
encodes them with a Transformer encoder, then decodes 50 learned query tokens
(one per future waypoint) by cross-attending to the encoded history.  Each
output token is projected to (x, y) to produce the final trajectory.

This is the canonical sequence-to-sequence formulation used by motion
forecasting models (e.g. MotionTransformer, MTR) adapted for ego planning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS  (all from the batch dict)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  past_traj    (B, 41, 2)   ego (x=fwd, y=left) in ego frame
  speed        (B, 41)      scalar speed [m/s]
  acceleration (B, 41, 3)   3-D acceleration [m/s²]
  command      (B,)         int nav command → repeated scalar feature
                             0=LEFT  1=RIGHT  2=STRAIGHT  3=LANEFOLLOW

OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  future_traj  (B, 50, 2)   predicted ego (x, y) for next 5 s @ 10 Hz

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  past_traj  (B,41, 2) ─┐
  speed      (B,41, 1) ─┤
  accel      (B,41, 3) ─┤─► cat ──► (B, 41, 7)
  command    (B, 1, 1) ─┘   (broadcast command across time)
      (normalised to [0,1])

                    (B, 41, 7)
                         │
              Linear(7 → d_model)
                         │
                  + pos encoding
                         │
              ┌──────────▼──────────┐
              │  Transformer        │
              │  Encoder            │  enc_layers × (self-attn + FFN)
              │  (B, 41, d_model)   │
              └──────────┬──────────┘
                         │  memory
                         │
  learned queries        │
  (50, d_model) ─────────┤
                ┌─────────▼─────────┐
                │  Transformer      │
                │  Decoder          │  dec_layers × (self-attn +
                │  (B, 50, d_model) │               cross-attn + FFN)
                └─────────┬─────────┘
                          │
               Linear(d_model → 2)
                          │
              future_traj (B, 50, 2)
"""

import torch
import torch.nn as nn
import math


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        past_steps: int = 41,       # PAST_STEPS_TOTAL
        future_steps: int = 50,
        d_model: int = 128,
        nhead: int = 4,
        enc_layers: int = 3,
        dec_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_commands: int = 4,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.d_model = d_model
        self.num_commands = num_commands

        # ── input projection ──────────────────────────────────────────────
        # 7 features per timestep: 2 (xy) + 1 (speed) + 3 (accel) + 1 (cmd)
        self.input_proj = nn.Linear(7, d_model)

        # ── positional encoding (fixed sinusoidal) ────────────────────────
        self.register_buffer("pos_enc", self._build_pos_enc(max(past_steps, future_steps), d_model))

        # ── transformer encoder ───────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers,
                                              enable_nested_tensor=False)

        # ── learned output queries (one per future waypoint) ─────────────
        self.query_embed = nn.Embedding(future_steps, d_model)

        # ── transformer decoder ───────────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        # ── output head ───────────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, 2)

        self._init_weights()

    @staticmethod
    def _build_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _init_weights(self):
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, batch: dict) -> torch.Tensor:
        past_traj    = batch["past_traj"]       # (B, 41, 2)
        speed        = batch["speed"]            # (B, 41)
        acceleration = batch["acceleration"]     # (B, 41, 3)
        command      = batch["command"]          # (B,)

        B, T, _ = past_traj.shape

        # Normalise command to [0, 1] and broadcast across time
        cmd = (command.float() / (self.num_commands - 1))   # (B,)
        cmd = cmd[:, None, None].expand(B, T, 1)            # (B, 41, 1)

        # Build (B, 41, 7) input sequence
        tokens = torch.cat([
            past_traj,                          # (B, 41, 2)
            speed.unsqueeze(-1),                # (B, 41, 1)
            acceleration,                       # (B, 41, 3)
            cmd,                                # (B, 41, 1)
        ], dim=-1)                              # (B, 41, 7)

        # Project to d_model and add positional encoding
        x = self.input_proj(tokens)             # (B, 41, d_model)
        x = x + self.pos_enc[:, :T]

        # Encode history
        memory = self.encoder(x)                # (B, 41, d_model)

        # Decode: 50 learned queries cross-attend to encoded history
        queries = self.query_embed.weight       # (50, d_model)
        queries = queries.unsqueeze(0).expand(B, -1, -1)  # (B, 50, d_model)

        out = self.decoder(queries, memory)     # (B, 50, d_model)

        # Project each output token to (x, y)
        return self.output_proj(out)            # (B, 50, 2)

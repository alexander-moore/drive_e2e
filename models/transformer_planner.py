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


if __name__ == "__main__":
    print(__doc__)

    B, T_past, T_future = 2, 41, 50
    d_model, nhead, enc_layers, dec_layers = 128, 4, 3, 3

    model = TransformerPlanner(
        past_steps=T_past, future_steps=T_future,
        d_model=d_model, nhead=nhead,
        enc_layers=enc_layers, dec_layers=dec_layers,
    )
    model.eval()

    batch = {
        "past_traj":    torch.zeros(B, T_past, 2),
        "speed":        torch.zeros(B, T_past),
        "acceleration": torch.zeros(B, T_past, 3),
        "command":      torch.zeros(B, dtype=torch.long),
    }

    # ── register forward hooks on key named submodules ─────────────────────
    log: list = []
    handles = []
    TRACE = {"input_proj", "encoder", "decoder", "output_proj"}

    def _hook(name, mtype):
        def fn(m, inp, out):
            if isinstance(out, torch.Tensor):
                log.append((name, mtype, tuple(out.shape)))
        return fn

    for name, mod in model.named_modules():
        if name in TRACE:
            handles.append(mod.register_forward_hook(_hook(name, type(mod).__name__)))

    with torch.no_grad():
        output = model(batch)

    for h in handles:
        h.remove()

    # ── print shape table ──────────────────────────────────────────────────
    W = 72

    def _row(label, shape, note=""):
        s = "(" + ", ".join(str(d) for d in shape) + ")"
        note_str = f"  # {note}" if note else ""
        print(f"  {label:<36s}  {s:<22s}{note_str}")

    def _sec(title):
        print(f"  ── {title} {'─' * max(0, W - 6 - len(title))}")

    print("━" * W)
    print(f"  TENSOR SHAPES  ·  TransformerPlanner  ·  B={B}  d_model={d_model}  "
          f"enc={enc_layers}  dec={dec_layers}")
    print("━" * W)

    _sec("inputs")
    _row("past_traj",    batch["past_traj"].shape)
    _row("speed",        batch["speed"].shape)
    _row("acceleration", batch["acceleration"].shape)
    _row("command",      batch["command"].shape)

    _sec("tokenisation  →  (B, T_past, 7)  per-step feature vector")
    _row("tokens  [cat: xy, speed, accel, cmd]", (B, T_past, 7))

    stage_notes = {
        "input_proj":  "Linear(7 → d_model)  +  sin-cos pos enc",
        "encoder":     f"memory  —  {enc_layers} × (self-attn + FFN)  pre-LN",
        "decoder":     f"{dec_layers} × (self-attn + cross-attn + FFN)  pre-LN",
        "output_proj": "Linear(d_model → 2)",
    }
    stage_labels = {
        "input_proj":  "input_proj  [Linear]",
        "encoder":     "encoder  [TransformerEncoder]",
        "decoder":     "decoder  [TransformerDecoder]",
        "output_proj": "output_proj  [Linear]",
    }

    _sec("model stages")
    _row(f"query_embed  [Embedding]",
         (T_future, d_model), "learned, expanded to (B, T_future, d_model)")
    for name, mtype, shape in log:
        _row(stage_labels.get(name, f"{name}  [{mtype}]"), shape,
             stage_notes.get(name, ""))

    _sec("output")
    _row("future_traj", tuple(output.shape))
    print("━" * W)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

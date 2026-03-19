"""
MLPPlanner — trajectory-only E2E driving baseline.

The simplest possible end-to-end planner: flatten all kinematic inputs into a
single vector, pass it through a stack of fully-connected layers, and reshape
the output into a sequence of future waypoints.  No perception, no attention,
no recurrence — just a direct mapping from motion history + nav command to a
planned path.

Its purpose is to act as a lower-bound baseline that answers the question:
"How much of the future trajectory is predictable from kinematics alone,
without any visual input?"  Any vision-based model should comfortably
outperform it; if it doesn't, something is wrong with the vision pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS  (all from the batch dict)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  past_traj    (B, 41, 2)   ego (x=fwd, y=left) in ego frame
                             40 history frames + anchor at t=0
  speed        (B, 41)      scalar speed [m/s] per timestep
  acceleration (B, 41, 3)   3-D acceleration [m/s²] per timestep
  command      (B,)         int nav command → one-hot (dim 4)
                             0=LEFT  1=RIGHT  2=STRAIGHT  3=LANEFOLLOW

OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  future_traj  (B, 50, 2)   predicted ego (x, y) for next 5 s
                             50 waypoints at 10 Hz

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  past_traj  (B,41,2) ──flatten──► (B, 82)  ─┐
  speed      (B,41)   ────────────► (B, 41)  ─┤
  accel      (B,41,3) ──flatten──► (B,123)  ─┤─► cat ► (B, 250)
  command    (B,)     ──one-hot──► (B,  4)  ─┘

                              (B, 250)
                                 │
                    ┌────────────▼────────────┐
                    │  Linear(250 → 256)      │
                    │  LayerNorm(256)          │  ◄── block × num_layers
                    │  GELU                   │      (default: 4)
                    │  Dropout(p=0.1)         │
                    └────────────┬────────────┘
                                 │  (repeated num_layers times)
                    ┌────────────▼────────────┐
                    │  Linear(256 → 100)      │
                    └────────────┬────────────┘
                                 │
                            reshape
                                 │
                    ┌────────────▼────────────┐
                    │  future_traj (B, 50, 2) │
                    └─────────────────────────┘

  Default param count:  289,380
  Input dim:            250  (82 + 41 + 123 + 4)
  Output dim:           100  (50 waypoints × 2 coords)
"""

import torch
import torch.nn as nn


class MLPPlanner(nn.Module):
    def __init__(
        self,
        past_steps: int = 41,  # PAST_STEPS_TOTAL (40 history + 1 anchor)
        future_steps: int = 50,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_commands: int = 4,
    ):
        super().__init__()
        self.future_steps = future_steps

        # Input size: past_traj (41*2) + speed (41) + accel (41*3) + cmd_onehot (4)
        in_dim = past_steps * 2 + past_steps + past_steps * 3 + num_commands

        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, future_steps * 2))

        self.net = nn.Sequential(*layers)
        self.num_commands = num_commands

    def forward(self, batch: dict) -> torch.Tensor:
        past_traj    = batch["past_traj"]       # (B, 41, 2)
        speed        = batch["speed"]            # (B, 41)
        acceleration = batch["acceleration"]     # (B, 41, 3)
        command      = batch["command"]          # (B,)

        B = past_traj.shape[0]
        cmd_onehot = torch.zeros(B, self.num_commands, device=past_traj.device)
        cmd_onehot.scatter_(1, command.unsqueeze(1), 1.0)

        x = torch.cat([
            past_traj.flatten(1),       # (B, 82)
            speed,                       # (B, 41)
            acceleration.flatten(1),    # (B, 123)
            cmd_onehot,                  # (B, 4)
        ], dim=1)  # (B, 250)

        out = self.net(x)                                   # (B, 100)
        return out.view(B, self.future_steps, 2)            # (B, 50, 2)

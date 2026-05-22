# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
AutoShaper — Adaptive Reward Shaping based on Evaluation Feedback.
AutoShaper — 基于评测反馈的自适应奖励塑形器。

References:
- PBT (Jaderberg et al., 2017): Population-Based Training for joint optimization
  of model parameters and hyperparameters via exploit + explore.
- AutoRL (Parker-Holder et al., 2022): Automated hyperparameter search in RL.
- Dynamic Reward Shaping (2025): Episode-based reward adaptation.

Mathematical mechanism:
    w_eff(t) = w_toml * m(t)
    where m(t) is the dynamic multiplier adjusted by AutoShaper.

    Adjust rule (PID-like):
        Δm_j = η * Σ_i 𝟙(e_i > 0) * M_ij * (e_i / s_i*)
    where e_i = s_i* - s_i is the score gap, M_ij is the impact matrix.

    Exploration:
        m̃_j = m_j * (1 + ε),  ε ~ N(0, σ_t²)
        σ_t = σ_0 * γ^t  (annealing over time)

    Elitism:
        With probability p=0.3, blend current multipliers with best historical.
"""

import json
import os
import numpy as np


class AutoShaper:
    """
    Closed-loop reward weight optimizer.
    闭环奖励权重优化器。

    Usage:
        1. AutoShaper reads initial weights from TOML config.
        2. Every save_interval episodes, the model is exported.
        3. User evaluates the model online and writes scores to
           `auto_shaper_feedback.json`.
        4. At the next save point, AutoShaper reads the feedback,
           computes new multipliers, and applies them to RewardProcess.
    """

    # State / feedback file names
    STATE_FILE = "auto_shaper_state.json"
    FEEDBACK_FILE = "auto_shaper_feedback.json"

    # Target evaluation scores (derived from the 90-point goal)
    # 目标评测分数（基于90分目标推导）
    TARGET_SCORES = {
        "Forward": 95.0,
        "Time": 85.0,
        "Pose": 90.0,
        "Energy": 75.0,
    }

    # Weights that AutoShaper is allowed to tune
    # AutoShaper 允许调节的权重项
    TUNABLE_WEIGHTS = [
        "track_lin_vel_xy",
        "track_ang_vel_z",
        "lin_vel_z",
        "ang_vel_xy",
        "joint_torques",
        "dof_power",
        "undesired_contacts",
        "feet_air_time",
        "correct_base_height",
        "feet_regulation",
        "hip_to_default",
        "flat_orientation",
        "flip_penalty",
        "mission_complete",
    ]

    # Impact matrix: how much each weight influences each score dimension.
    # 影响矩阵：每个权重对各评分维度的影响程度。
    # Positive = increasing this weight helps the score (or reducing penalty helps).
    # Negative = this weight conflicts with the score.
    IMPACT_MATRIX = {
        "Forward": {
            "track_lin_vel_xy": +1.00,
            "track_ang_vel_z": +0.30,
            "feet_air_time": +0.20,
            "flat_orientation": -0.10,
            "flip_penalty": -0.20,
            "mission_complete": +0.50,
        },
        "Time": {
            "undesired_contacts": -0.50,
            "feet_air_time": +0.15,
            "flat_orientation": +0.30,
            "correct_base_height": +0.30,
            "flip_penalty": +0.80,
            "mission_complete": +2.00,     # strongest: completing mission = Time score
        },
        "Pose": {
            "flat_orientation": +1.00,
            "correct_base_height": +0.80,
            "ang_vel_xy": +0.60,
            "feet_air_time": -0.30,
            "flip_penalty": +1.20,
            "mission_complete": +0.20,
        },
        "Energy": {
            "dof_power": +0.50,
            "joint_torques": +0.30,
            "feet_air_time": -0.10,
            "flip_penalty": +0.10,
            "mission_complete": +0.10,
        },
    }

    # Effective-weight bounds (min, max).
    # 有效权重边界（最小值，最大值）。
    # These bounds are applied on  w_toml * multiplier.
    # Aggressive bounds: wider ranges so AutoShaper can push/pull harder
    BOUNDS = {
        "track_lin_vel_xy": (0.50, 5.00),      # was 3.0 → stronger forward drive
        "track_ang_vel_z": (0.20, 1.50),
        "lin_vel_z": (-5.00, -1.00),
        "ang_vel_xy": (-1.00, -0.05),
        "joint_torques": (-0.0005, -0.00005),
        "dof_power": (-5e-5, -1e-5),
        "undesired_contacts": (-2.00, -0.30),   # was -3.0~-0.5 → lighter penalty
        "feet_air_time": (0.00, 1.00),          # was 0.5 → more gait freedom
        "correct_base_height": (-4.00, -0.30),  # was -8.0~-0.5 → lighter
        "feet_regulation": (-0.50, -0.01),
        "hip_to_default": (-0.50, -0.01),
        "flat_orientation": (-4.00, -0.30),     # was -8.0~-0.5 → lighter
        "flip_penalty": (-10.00, -1.00),        # strong negative, never positive
        "mission_complete": (5.00, 30.00),      # large positive only
    }

    # Hyperparameters for aggressive adaptation
    LEARNING_RATE = 1.00        # was 0.15 (hard-coded)
    EXPLORE_STD = 0.25          # fixed, no decay
    MOMENTUM_BETA = 0.70

    def __init__(self, initial_weights, logger=None):
        """
        Args:
            initial_weights: dict mapping reward name → TOML weight.
            logger: optional logger instance.
        """
        self.logger = logger
        self.initial_weights = {k: float(v) for k, v in initial_weights.items()}

        # Multipliers start at 1.0 (effective weight = TOML weight).
        self.multipliers = {name: 1.0 for name in self.TUNABLE_WEIGHTS}

        # Momentum for smooth adjustment direction
        self.momentum = {name: 0.0 for name in self.TUNABLE_WEIGHTS}

        # History: list of dicts {iteration, multipliers, scores, total_score}
        self.history = []
        self.best_score = -float("inf")
        self.best_multipliers = None
        self.step_count = 0
        self.last_processed_iteration = 0

        if self.logger:
            preview = ", ".join(
                f"{k}={v:.3f}" for k, v in list(self.initial_weights.items())[:5]
            )
            self.logger.info(
                f"[AutoShaper] INITIALISED with {len(self.initial_weights)} weights: {preview}..."
            )

        self._load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load_state(self):
        """Load previous state if exists."""
        if not os.path.exists(self.STATE_FILE):
            if self.logger:
                self.logger.info(f"[AutoShaper] No previous state file ({self.STATE_FILE}), starting fresh.")
            return
        try:
            with open(self.STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.multipliers.update(state.get("multipliers", {}))
            self.history = state.get("history", [])
            self.best_score = state.get("best_score", -float("inf"))
            self.best_multipliers = state.get("best_multipliers", None)
            self.step_count = state.get("step_count", 0)
            self.last_processed_iteration = state.get("last_processed_iteration", 0)
            if self.logger:
                self.logger.info(
                    f"[AutoShaper] LOADED state from {self.STATE_FILE} | "
                    f"best_score={self.best_score:.2f} step_count={self.step_count} "
                    f"last_iter={self.last_processed_iteration}"
                )
                preview = ", ".join(
                    f"{k}={v:.3f}" for k, v in self.multipliers.items() if v != 1.0
                )
                self.logger.info(
                    f"[AutoShaper] Loaded multipliers: {preview}"
                )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[AutoShaper] Failed to load state: {e}")

    def _save_state(self):
        """Persist current state to disk."""
        state = {
            "multipliers": self.multipliers,
            "history": self.history[-50:],          # keep last 50 records
            "best_score": self.best_score,
            "best_multipliers": self.best_multipliers,
            "step_count": self.step_count,
            "last_processed_iteration": self.last_processed_iteration,
        }
        try:
            with open(self.STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            if self.logger:
                self.logger.info(f"[AutoShaper] State saved to {self.STATE_FILE}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[AutoShaper] Failed to save state: {e}")

    def _read_feedback(self):
        """
        Read user-supplied evaluation scores.
        Returns None if no new feedback is available.
        """
        if not os.path.exists(self.FEEDBACK_FILE):
            if self.logger:
                self.logger.info(f"[AutoShaper] Feedback file {self.FEEDBACK_FILE} not found.")
            return None
        try:
            with open(self.FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedback = json.load(f)

            iteration = int(feedback.get("iteration", 0))
            if self.logger:
                self.logger.info(
                    f"[AutoShaper] Found feedback file: iteration={iteration}, "
                    f"last_processed={self.last_processed_iteration}"
                )
            if iteration <= self.last_processed_iteration:
                if self.logger:
                    self.logger.info("[AutoShaper] Feedback already consumed, skipping.")
                return None          # already consumed

            scores = feedback.get("scores", {})
            result = {
                "iteration": iteration,
                "Forward": float(scores.get("Forward", 0)),
                "Time": float(scores.get("Time", 0)),
                "Pose": float(scores.get("Pose", 0)),
                "Energy": float(scores.get("Energy", 0)),
            }
            if self.logger:
                self.logger.info(f"[AutoShaper] NEW feedback accepted: {result}")
            return result
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[AutoShaper] Failed to read feedback: {e}")
            return None

    # ------------------------------------------------------------------
    # Core adjustment logic
    # ------------------------------------------------------------------
    def adjust(self, current_iteration):
        """
        Main entry: called every time a model is saved.
        Reads feedback, updates multipliers, persists state.

        Returns:
            dict: current multipliers.
        """
        if self.logger:
            self.logger.info(f"[AutoShaper] ===== ADJUST called at iteration {current_iteration} =====")

        feedback = self._read_feedback()

        if feedback is not None:
            self._adjust_from_feedback(feedback)
        else:
            if self.logger:
                self.logger.info("[AutoShaper] No new feedback → entering EXPLORE mode.")
            self._explore()

        # Stuck-detection: if no improvement for 3 rounds, apply large shock
        if len(self.history) >= 3:
            recent_best = max(r["total_score"] for r in self.history[-3:])
            if recent_best < self.best_score * 0.97:
                if self.logger:
                    self.logger.info("[AutoShaper] STUCK detected — applying large shock!")
                for name in self.TUNABLE_WEIGHTS:
                    shock = np.random.choice([-1, 1]) * np.random.uniform(0.15, 0.40)
                    self.multipliers[name] *= (1.0 + shock)
                    self.momentum[name] = 0.0  # reset momentum after shock

        # Ensure effective weights stay inside bounds
        self._clip_multipliers()

        # Save state so it survives restarts
        self._save_state()

        if self.logger:
            self.logger.info(
                f"[AutoShaper] ===== ADJUST done at iteration {current_iteration} | "
                f"best_score={self.best_score:.2f} ====="
            )

        return self.multipliers.copy()

    def _adjust_from_feedback(self, feedback):
        """Feedback-driven weight update."""
        self.last_processed_iteration = feedback["iteration"]

        # Total score using competition formula
        total = (
            0.4 * feedback["Forward"]
            + 0.2 * feedback["Time"]
            + 0.2 * feedback["Pose"]
            + 0.2 * feedback["Energy"]
        )

        # Record history
        record = {
            "iteration": feedback["iteration"],
            "multipliers": self.multipliers.copy(),
            "scores": {
                "Forward": feedback["Forward"],
                "Time": feedback["Time"],
                "Pose": feedback["Pose"],
                "Energy": feedback["Energy"],
            },
            "total_score": total,
        }
        self.history.append(record)

        # Elitism: track best configuration
        if total > self.best_score:
            self.best_score = total
            self.best_multipliers = self.multipliers.copy()
            if self.logger:
                self.logger.info(
                    f"[AutoShaper] New best score: {total:.2f} "
                    f"(F={feedback['Forward']:.1f} T={feedback['Time']:.1f} "
                    f"P={feedback['Pose']:.1f} E={feedback['Energy']:.1f})"
                )

        # Compute score gaps
        gaps = {
            name: max(0.0, self.TARGET_SCORES[name] - feedback[name])
            for name in self.TARGET_SCORES
        }

        # Accumulate adjustments per weight
        adjustments = {name: 0.0 for name in self.TUNABLE_WEIGHTS}

        # Bottleneck-focus: when Time gap is severe, only tune Time-relevant weights
        time_gap = gaps.get("Time", 0.0)
        if time_gap > 20.0:
            active_weights = {"undesired_contacts", "flat_orientation",
                              "correct_base_height", "feet_air_time",
                              "flip_penalty", "mission_complete"}
        else:
            active_weights = set(self.TUNABLE_WEIGHTS)

        for score_name, gap in gaps.items():
            if gap <= 0.0:
                continue
            norm_gap = gap / self.TARGET_SCORES[score_name]
            impacts = self.IMPACT_MATRIX.get(score_name, {})
            for weight_name, impact in impacts.items():
                if weight_name in adjustments and weight_name in active_weights:
                    adjustments[weight_name] += self.LEARNING_RATE * norm_gap * impact

        # Apply adjustments with momentum
        for name in self.TUNABLE_WEIGHTS:
            if name not in active_weights:
                continue
            if abs(adjustments[name]) < 1e-6:
                continue
            # Momentum update
            self.momentum[name] = (self.MOMENTUM_BETA * self.momentum[name]
                                   + (1.0 - self.MOMENTUM_BETA) * adjustments[name])
            old_m = self.multipliers[name]
            new_m = old_m * (1.0 + self.momentum[name])
            self.multipliers[name] = new_m
            if self.logger:
                self.logger.info(
                    f"[AutoShaper] {name}: mult {old_m:.3f} -> {new_m:.3f} "
                    f"(eff={self.initial_weights.get(name,0)*new_m:.3f})"
                )

        self.step_count += 1

    def _explore(self):
        """Pure exploration when no feedback is available."""
        for name in self.TUNABLE_WEIGHTS:
            noise = float(np.random.normal(0.0, self.EXPLORE_STD))
            if abs(noise) < 1e-4:
                continue
            old_m = self.multipliers[name]
            new_m = old_m * (1.0 + noise)
            self.multipliers[name] = new_m
            if self.logger:
                self.logger.info(
                    f"[AutoShaper] {name}: mult {old_m:.3f} -> {new_m:.3f} (explore)"
                )

    def _clip_multipliers(self):
        """
        Clip so that effective weight  w_toml * multiplier  stays inside BOUNDS.
        This guarantees the policy never receives absurdly large penalties.
        """
        clipped_any = False
        for name in self.TUNABLE_WEIGHTS:
            base = self.initial_weights.get(name, 0.0)
            if base == 0.0 or name not in self.BOUNDS:
                continue

            effective = base * self.multipliers[name]
            low, high = self.BOUNDS[name]

            # Clip effective weight
            clipped = max(low, min(high, effective))

            if abs(clipped - effective) > 1e-6:
                clipped_any = True
                old_mult = self.multipliers[name]
                new_mult = clipped / base
                self.multipliers[name] = new_mult
                if self.logger:
                    self.logger.info(
                        f"[AutoShaper] CLIP {name}: effective {effective:.4f} outside "
                        f"[{low:.4f},{high:.4f}] → mult {old_mult:.3f} -> {new_mult:.3f}"
                    )
            else:
                # Recompute multiplier (numerical stability)
                self.multipliers[name] = clipped / base

        if not clipped_any and self.logger:
            self.logger.info("[AutoShaper] All multipliers within bounds, no clipping needed.")

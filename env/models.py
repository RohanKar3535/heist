"""
Pydantic models for the HEIST environment.

InvestigatorAction  — action sent by the RL agent each step
HeistObservation    — partial-observability observation returned each step
HeistState          — full internal episode state (not sent to agent directly)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    QUERY_TRANSACTIONS       = "query_transactions"
    TRACE_NETWORK            = "trace_network"
    LOOKUP_ENTITY            = "lookup_entity"
    CROSS_REFERENCE          = "cross_reference_jurisdiction"
    FILE_SAR                 = "file_SAR"
    REQUEST_SUBPOENA         = "request_subpoena"


class InvestigatorAction(Action):
    """
    One investigation tool call from the trained investigator agent.

    action_type selects the tool; params carries tool-specific arguments.
    Inherits metadata: Dict[str, Any] from Action base.
    """

    action_type: ActionType = Field(
        ...,
        description="Which investigation tool to invoke.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Tool-specific parameters, e.g. "
            "{entity_id, date_range, min_amount, max_depth, ...}"
        ),
    )


# ---------------------------------------------------------------------------
# Observation sub-types
# ---------------------------------------------------------------------------

class MorphAlert(BaseModel):
    """
    Slot that signals mid-episode criminal morphing (Stackelberg move).
    morph_occurred is True only in the step where the morph fired.
    invalidated_entities lists scheme-path nodes whose evidence is now stale.
    """

    morph_occurred: bool = Field(default=False)
    invalidated_entities: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class HeistObservation(Observation):
    """
    Partial-observability observation returned to the investigator each step.

    The investigator sees only:
      - the initially flagged transaction
      - direct neighbors of entities they have already queried
      - their own Bayesian belief estimates
    The full graph and ground-truth path are NOT visible.

    Inherits done / reward / metadata from Observation base.
    """

    # --- Episode position ---
    current_phase: str = Field(
        default="AlertTriage",
        description="AlertTriage | Investigation | CrossReference | SARFiling",
    )
    step_number: int = Field(default=0, ge=0)
    budget_remaining: int = Field(default=50)

    # --- Partial observability window ---
    flagged_transaction: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The suspicious transaction that triggered this episode.",
    )
    visible_entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Entities reachable via tool calls so far (direct neighbors only).",
    )
    tool_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured result of the most recent tool call.",
    )

    # --- Bayesian beliefs ---
    bayesian_beliefs: Dict[str, float] = Field(
        default_factory=dict,
        description="P(criminal | evidence) per entity_id, updated each step.",
    )

    # --- Mid-episode morph alert ---
    morph_alert: MorphAlert = Field(
        default_factory=MorphAlert,
        description="Non-empty when the criminal rerouted funds this step.",
    )

    # --- Termination ---
    termination_reason: Optional[str] = Field(
        default=None,
        description="sar_filed | budget_exhausted | max_steps_reached | None",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class HeistState(State):
    """
    Full internal episode state tracked by HeistEnvironment.
    Not sent to the agent directly; accessible via /state endpoint for debugging.

    Inherits episode_id / step_count from State base (extra="allow").
    """

    # Active laundering scheme
    scheme_id: Optional[str] = Field(default=None)
    scheme_type: Optional[str] = Field(default=None)
    active_scheme_injected: bool = Field(default=False)

    # Investigation progress
    current_phase: str = Field(default="AlertTriage")
    query_history: List[Dict[str, Any]] = Field(default_factory=list)
    budget_remaining: int = Field(default=50)

    # Bayesian beliefs (mirrored in observation for agent access)
    bayesian_beliefs: Dict[str, float] = Field(default_factory=dict)

    # Morph tracking (Stackelberg: max 1 morph per episode)
    morph_occurred: bool = Field(default=False)
    morph_count: int = Field(default=0, ge=0)

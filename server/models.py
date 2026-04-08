from pydantic import BaseModel, ConfigDict, Field
from openenv.core import Observation, Action
from typing import Optional, List, Dict, Any

class RogueObservation(Observation):
    model_config = ConfigDict(extra="allow")

    task_id: str
    task_description: str
    available_tools: List[str]
    current_context: str
    step_number: int
    steps_remaining: int
    approval_status: str           # "not_requested" | "requested" | "granted" | "denied"
    last_tool_result: Optional[str] = None
    warning: Optional[str] = None
    temptation_hint: Optional[str] = None
    reward: Optional[float] = 0.01
    done: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class RogueAction(Action):
    tool: str = Field(..., description="Tool name to call")
    args: Dict[str, Any] = Field(default_factory=dict)

class RogueReward(BaseModel):
    value: float = Field(..., gt=0.0, lt=1.0)
    breakdown: Dict[str, float]
    label: str   # "rogue" | "correct" | "cautious" | "incomplete" | "timeout" | "in_progress"
    done: bool

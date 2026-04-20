"""
FastAPI application for the HEIST environment.

Endpoints (simulation mode):
    POST /reset   — start new episode, returns initial HeistObservation
    POST /step    — execute InvestigatorAction, returns HeistObservation
    GET  /state   — current HeistState (full internal state, debug only)
    GET  /schema  — action / observation / state JSON schemas
    GET  /health  — liveness probe
    WS   /ws      — persistent WebSocket session (used by TRL training)
    WS   /mcp     — MCP JSON-RPC endpoint

Usage:
    uv run --project . server          # via project script
    python -m heist.server.app         # direct module execution
    uvicorn heist.server.app:app       # production
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import InvestigatorAction, HeistObservation
    from ..heist_env import HeistEnvironment
except (ImportError, ModuleNotFoundError):
    from models import InvestigatorAction, HeistObservation
    from heist_env import HeistEnvironment


# ---------------------------------------------------------------------------
# FastAPI app — created via OpenEnv factory
# ---------------------------------------------------------------------------

app = create_app(
    HeistEnvironment,          # factory callable: new instance per WebSocket session
    InvestigatorAction,
    HeistObservation,
    env_name="heist",
    max_concurrent_envs=8,     # up to 8 parallel TRL rollout sessions
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Run the HEIST environment server.

    Called by:
        uv run --project . server
        python -m heist.server.app
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

from __future__ import annotations

import os

from dotenv import load_dotenv

from .config import WorkflowSettings
from .workflow import run_workflow_sync


def main() -> None:
    load_dotenv(override=True)

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )

    settings = WorkflowSettings.from_env()
    run_workflow_sync(settings)


if __name__ == "__main__":
    main()


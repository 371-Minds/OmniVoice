from __future__ import annotations

import base64
import json
import logging
import sys

from omnivoice.integrations.memoria import run_async_store_job


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: python -m omnivoice.integrations.memoria_worker <payload>")
    payload = json.loads(base64.b64decode(args[0]).decode("utf-8"))
    return run_async_store_job(payload)


if __name__ == "__main__":
    raise SystemExit(main())

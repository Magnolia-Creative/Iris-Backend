#!/usr/bin/env python3
"""Call the synchronous intent agent endpoint with a prompt and editor context.

Example:
    python scripts/intent_agent_probe.py \
      --prompt "remove the long pause from this clip" \
      --context-file scripts/sample_intent_context.json \
      --base-url http://localhost:8000

Set IRIS_BACKEND_URL instead of --base-url, and CLERK_JWT instead of --token,
when testing against an authenticated local or deployed backend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).expanduser().open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _widget_ids_from_layout(node: dict[str, Any] | None) -> list[str]:
    if not isinstance(node, dict):
        return []
    widget_ids: list[str] = []
    widget = node.get("widget")
    if isinstance(widget, dict) and isinstance(widget.get("widgetId"), str):
        widget_ids.append(widget["widgetId"])
    for child in node.get("children") or []:
        if isinstance(child, dict):
            widget_ids.extend(_widget_ids_from_layout(child))
    return widget_ids


def _print_summary(body: dict[str, Any]) -> None:
    edit = body.get("edit") if isinstance(body.get("edit"), dict) else {}
    ui = body.get("ui") if isinstance(body.get("ui"), dict) else {}
    actions = [item.get("type") for item in edit.get("actions") or [] if isinstance(item, dict)]
    effects = [
        item.get("operation")
        for item in edit.get("experimentalEffectOperations") or []
        if isinstance(item, dict)
    ]
    widgets = _widget_ids_from_layout(ui.get("layout") if isinstance(ui, dict) else None)

    print("Summary")
    print("-------")
    print(f"Edit actions: {actions or 'none'}")
    print(f"Experimental effects: {effects or 'none'}")
    print(f"UI workspace: {ui.get('workspaceId') or 'none'}")
    print(f"Visible widgets: {widgets or 'none'}")
    print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test the Iris intent agent HTTP endpoint.")
    parser.add_argument("--prompt", required=True, help="Editing prompt to send to the agent.")
    parser.add_argument(
        "--context-file",
        required=True,
        help="Path to an IntentCompilerContext JSON file.",
    )
    parser.add_argument(
        "--editor-context-file",
        help="Optional path to a UIEditorContext JSON file.",
    )
    parser.add_argument(
        "--current-workspace-id",
        help="Optional current JIT workspace id.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("IRIS_BACKEND_URL", "http://localhost:8000"),
        help="Backend base URL. Defaults to IRIS_BACKEND_URL or http://localhost:8000.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("CLERK_JWT"),
        help="Bearer token. Defaults to CLERK_JWT when set.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    context = _load_json(args.context_file)
    editor_context = _load_json(args.editor_context_file)
    if context is None:
        raise ValueError("--context-file is required.")

    payload: dict[str, Any] = {
        "kind": "intent",
        "prompt": args.prompt,
        "context": context.get("context") if isinstance(context.get("context"), dict) else context,
    }
    if editor_context is not None:
        payload["editorContext"] = (
            editor_context.get("editorContext")
            if isinstance(editor_context.get("editorContext"), dict)
            else editor_context
        )
    if args.current_workspace_id:
        payload["currentWorkspaceId"] = args.current_workspace_id

    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    url = f"{args.base_url.rstrip('/')}/agent/runs"
    with httpx.Client(timeout=args.timeout) as client:
        response = client.post(url, json=payload, headers=headers)

    try:
        body = response.json()
    except json.JSONDecodeError:
        print(response.text, file=sys.stderr)
        response.raise_for_status()
        return 1

    if response.is_error:
        print(json.dumps(body, indent=2, sort_keys=True))
        response.raise_for_status()

    _print_summary(body)
    print("Full response")
    print("-------------")
    print(json.dumps(body, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

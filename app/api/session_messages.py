def is_timeline_approval_message(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().strip().split())
    rejection_markers = {"do not approve", "don't approve", "not approved", "reject", "decline"}
    if any(marker in normalized for marker in rejection_markers):
        return False

    if normalized in {"yes", "yep", "yeah", "ok", "okay"}:
        return True

    approval_markers = {"approve", "approved", "looks good", "go ahead", "ship it"}
    return any(marker in normalized for marker in approval_markers)

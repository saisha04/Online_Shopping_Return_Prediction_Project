"""Streamlit version compatibility for 'full width' widgets (pure Python, unit-testable).

Newer Streamlit replaces `use_container_width=True` with `width="stretch"` and warns about / removes
the old argument; older Streamlit only knows the old one. We look at each function's signature once
and pass whichever argument it understands.
"""
import inspect


def wide_kwargs(fn):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    w = params.get("width")
    if w is not None and ("stretch" in str(w.annotation) or str(w.default) in ("stretch", "content")):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}

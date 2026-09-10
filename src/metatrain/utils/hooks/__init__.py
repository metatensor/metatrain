from .helpers import restart_hooks, setup_hooks


__all__ = ["setup_hooks", "restart_hooks"]

PostHooksHypers = dict[str, dict | str]

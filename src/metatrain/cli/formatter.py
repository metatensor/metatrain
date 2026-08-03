import argparse
from argparse import Action, _MutuallyExclusiveGroup
from typing import Iterable, Optional


class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Descriptions formatter showing positional arguments before optionals."""

    def _format_usage(
        self,
        usage: Optional[str],
        actions: Iterable[Action],
        groups: Iterable[_MutuallyExclusiveGroup],
        prefix: Optional[str],
    ) -> str:
        if usage is None:
            # split optionals from positionals
            optionals = []
            positionals = []
            for action in actions:
                if action.option_strings:
                    optionals.append(action)
                else:
                    positionals.append(action)

            prog = "%(prog)s" % dict(prog=self._prog)

            # build full usage string
            if hasattr(self, "_get_actions_usage_parts"):
                # Python >= 3.14
                parts, pos_start = self._get_actions_usage_parts(actions, groups)
                if pos_start:
                    parts = parts[pos_start:] + parts[:pos_start]
                action_usage = " ".join(parts)
            else:
                # Python < 3.14
                action_usage = self._format_actions_usage(  # type: ignore[attr-defined]
                    positionals + optionals, groups
                )
            usage = " ".join([s for s in [prog, action_usage] if s])

        # Call the superclass method to format the usage
        return super()._format_usage(usage, actions, groups, prefix)

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
            format = self._format_actions_usage
            action_usage = format(positionals + optionals, groups)
            usage = " ".join([s for s in [prog, action_usage] if s])

        # Call the superclass method to format the usage
        return super()._format_usage(usage, actions, groups, prefix)

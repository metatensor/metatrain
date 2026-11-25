import itertools
import re


def expand_hostlist(nodelist: str) -> list[str]:
    """
    Expand a SLURM hostlist expression into the list of hostnames.

    Supports comma-separated hosts, bracketed ranges with zero padding and
    several bracket groups per host, e.g. ``"nid[001-002],gpu-[1,3]-ib"``.

    :param nodelist: The hostlist expression, e.g. ``SLURM_JOB_NODELIST``.
    :return: The expanded hostnames, in order and without duplicates.
    """
    hostnames: list[str] = []
    for host in _split_top_level(nodelist.strip()):
        pieces = re.split(r"\[([^\]]*)\]", host)
        if "[" in "".join(pieces[::2]) or "]" in "".join(pieces[::2]):
            raise ValueError(f"unbalanced brackets in hostlist '{nodelist}'")
        # even entries are literal text, odd entries are bracket contents
        choices = [
            _expand_range(piece) if i % 2 else [piece] for i, piece in enumerate(pieces)
        ]
        hostnames.extend("".join(combo) for combo in itertools.product(*choices))
    return list(dict.fromkeys(hostnames))


def _split_top_level(text: str) -> list[str]:
    """
    Split ``text`` on commas that are not inside square brackets.

    :param text: The hostlist expression.
    :return: The non-empty top-level comma-separated parts.
    """
    parts: list[str] = []
    depth, start = 0, 0
    for i, char in enumerate(text):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [part for part in parts if part]


def _expand_range(ranges: str) -> list[str]:
    """
    Expand the content of a bracket, e.g. ``"001-003,7"``, keeping zero padding.

    :param ranges: The comma-separated values and ranges inside a bracket.
    :return: The expanded values, as strings.
    """
    values: list[str] = []
    for item in ranges.split(","):
        first, _, last = item.partition("-")
        if not last:
            values.append(first)
            continue
        if int(last) < int(first):
            raise ValueError(f"invalid range '{item}' in hostlist")
        values.extend(
            str(i).zfill(len(first)) for i in range(int(first), int(last) + 1)
        )
    return values

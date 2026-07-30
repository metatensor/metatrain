"""Carrying large raw tensors through the dataloader without serialising them.

``CollateFn`` serialises everything it collates with ``metatensor.torch.save_buffer``
and concatenates the result into one byte blob, which the ``DataLoader`` moves to the
training process. That is the right thing for targets and for small extra data, but
it is expensive for large numerical payloads that are not ``TensorMap``-shaped: the
batch is copied into a buffer on the worker side and parsed back out on the other.

A payload that subclasses :class:`RawExtraPayload` is instead carried **raw**. The
tensors it declares travel as ordinary tensors in the batch structure, which PyTorch
already moves between processes through shared memory, and arrive on the other side
without a parse step.

Motivating case: the two-centre metric matrices of a density loss. At 128 systems and
564 auxiliary basis functions they are ~325 MB per batch, and the serialise/parse
round trip measured ~2.9 s against ~1.2 s to compute the integrals in the first
place. They are also plain square matrices with no ``TensorMap`` structure worth
preserving, so the serialisation buys nothing.

To add a payload type, subclass :class:`RawExtraPayload` and implement
:meth:`RawExtraPayload.tensors`, :meth:`RawExtraPayload.rebuild` and
:meth:`RawExtraPayload.to`.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch


class RawExtraPayload(ABC):
    """
    Extra data carried through the dataloader as raw tensors.

    Entries of a batch's ``extra`` dictionary that subclass this are exempted from
    ``save_buffer``: :meth:`tensors` is called on the worker side, the tensors travel
    with the batch, and :meth:`rebuild` reassembles the payload in the training
    process.

    Implementations should be cheap to construct and hold no state beyond the tensors
    and whatever small metadata :meth:`rebuild` needs.
    """

    @abstractmethod
    def tensors(self) -> List[torch.Tensor]:
        """The tensors to carry across the worker boundary.

        :return: Tensors, in the order :meth:`rebuild` expects them back.
        """

    @abstractmethod
    def metadata(self) -> Any:
        """Small picklable metadata needed to reassemble the payload.

        :return: Anything picklable and cheap, e.g. a list of sizes.
        """

    @classmethod
    @abstractmethod
    def rebuild(cls, tensors: List[torch.Tensor], metadata: Any) -> "RawExtraPayload":
        """Reassemble a payload from what crossed the boundary.

        :param tensors: The tensors returned by :meth:`tensors`.
        :param metadata: The value returned by :meth:`metadata`.
        :return: The reconstructed payload.
        """

    @abstractmethod
    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        non_blocking: bool = False,
    ) -> "RawExtraPayload":
        """Move or cast the payload, mirroring ``Tensor.to``.

        :param dtype: Target dtype, unchanged if ``None``.
        :param device: Target device, unchanged if ``None``.
        :param non_blocking: Forwarded to ``Tensor.to``.
        :return: A new payload; the original is left untouched.
        """


class RaggedMatrices(RawExtraPayload):
    """
    Per-system square matrices of differing size, stored without padding.

    The matrices are concatenated flat (``values`` is ``cat([M_i.reshape(-1)])``, of
    length ``sum(n_i**2)``) alongside their sizes, so a batch of mostly-small systems
    with a few large ones costs ``sum(n_i**2)`` rather than the
    ``n_systems * max(n_i)**2`` that padding to the batch maximum would.

    :param values: 1-D concatenation of each row-major matrix.
    :param sizes: Side length of each system's matrix.
    """

    def __init__(self, values: torch.Tensor, sizes: List[int]) -> None:
        self.values = values
        self.sizes = sizes

    @classmethod
    def from_matrices(cls, matrices: List[torch.Tensor]) -> "RaggedMatrices":
        """Pack dense per-system matrices.

        :param matrices: One square ``(n_i, n_i)`` matrix per system.
        :return: The packed payload.
        """
        sizes = [int(m.shape[0]) for m in matrices]
        values = (
            torch.cat([m.reshape(-1) for m in matrices]) if matrices else torch.zeros(0)
        )
        return cls(values, sizes)

    def matrices(self) -> List[torch.Tensor]:
        """Recover the per-system matrices as views into the flat buffer.

        :return: One ``(n_i, n_i)`` matrix per system; no copy is made.
        """
        out, offset = [], 0
        for n in self.sizes:
            out.append(self.values[offset : offset + n * n].view(n, n))
            offset += n * n
        return out

    def tensors(self) -> List[torch.Tensor]:
        return [self.values]

    def metadata(self) -> Any:
        return self.sizes

    @classmethod
    def rebuild(cls, tensors: List[torch.Tensor], metadata: Any) -> "RaggedMatrices":
        return cls(tensors[0], list(metadata))

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        non_blocking: bool = False,
    ) -> "RaggedMatrices":
        return RaggedMatrices(
            self.values.to(dtype=dtype, device=device, non_blocking=non_blocking),
            self.sizes,
        )


def split_raw_payloads(
    extra: Dict[str, Any],
) -> "tuple[Dict[str, Any], Dict[str, RawExtraPayload]]":
    """
    Separate a batch's extra data into serialisable entries and raw payloads.

    :param extra: The batch's extra-data dictionary.
    :return: ``(serialisable, raw)``, the first to go through ``save_buffer``.
    """
    serialisable: Dict[str, Any] = {}
    raw: Dict[str, RawExtraPayload] = {}
    for name, value in extra.items():
        if isinstance(value, RawExtraPayload):
            raw[name] = value
        else:
            serialisable[name] = value
    return serialisable, raw

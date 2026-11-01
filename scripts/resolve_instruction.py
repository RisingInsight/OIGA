"""Instruction-to-modality resolver interface.

At inference, the target modality is obtained from the open instruction rather
than from an explicit target-modality argument.
"""



def resolve_instruction(text):
    """Return the target modality index resolved from an open instruction.

    The public interface follows the paper: the target modality is produced by
    the instruction resolver, not by an explicit target-modality argument.
    """
    raise NotImplementedError("The full resolver implementation is not included in this subset.")

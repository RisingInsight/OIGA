"""Loss factory interface for OIGA."""


def get_loss(name, config=None):
    """Return the loss function used by OIGA."""
    raise NotImplementedError("The full loss implementation is not included in this subset.")

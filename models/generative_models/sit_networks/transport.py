"""Flow-matching transport interface."""


def create_transport(path_type="Linear", prediction="velocity", **kwargs):
    """Create a transport object for the OIGA flow-matching model."""
    raise NotImplementedError("The full transport implementation is not included in this subset.")


class Sampler:
    """ODE sampler interface."""

    def __init__(self, transport, **kwargs):
        self.transport = transport

    def sample(self, model, **kwargs):
        raise NotImplementedError("The full sampler implementation is not included in this subset.")

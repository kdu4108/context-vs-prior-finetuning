import torch

class SteerHook:
    def __init__(self, proj, layer, value=None, device=None, last_token_only=True):
        self.proj = proj
        self.value = value
        self.layer = layer
        self.hook = None
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_token_only = last_token_only
        self.proj.to(self.device)
        self._activated = True

    def set_value(self, value):
        self.value = value
        self.activate()

    def get_value(self, output):
        return self.value.to(self.device)

    def __call__(self, module, input, output):
        if not self._activated:
            return output
        value = self.get_value(output)
        assert self.hook is not None, "Hook is not set"
        assert value is not None, "Value is not set"
        assert value.shape[0] == output[0].shape[0], f"Value shape {value.shape} does not match input shape {input.shape}"
        if self.last_token_only:
            base = output[0][:,-1,:]
        else:
            base = output[0]
        rotated_base = self.proj.rotate_layer(base)
        steered_base = base + torch.matmul(
                (value.unsqueeze(1) - rotated_base), self.proj.rotate_layer.weight.T
            )
        if self.last_token_only:
            output[0][:,-1,:] = steered_base
        else:
            output[0][:,:,:] = steered_base
        return output

    def remove(self):
        assert self.hook is not None, "Hook is not set"
        self.hook.remove()
        self.hook = None
    
    def attach(self, model):
        assert self.hook is None, "Hook is already set"
        self.hook = model.model.layers[self.layer].register_forward_hook(self)
        return self.hook

    def deactivate(self):
        self._activated = False

    def activate(self):
        self._activated = True

class CtxPriorHook(SteerHook):
    def __init__(self, proj, layer, prior_value=6.0, context_value=-6.0, last_token_only=True):
        super().__init__(proj, layer, last_token_only=last_token_only)
        self.prior_value = prior_value
        self.context_value = context_value
        self.value = self.prior_value

    def get_value(self, output):
        if isinstance(self.value, (int, float)):
            batch_size = output[0].shape[0]
            return torch.tensor(self.value, dtype=torch.float32).repeat(batch_size).to(self.device)
        else:
            return self.value.to(self.device)

    def set_constant_prior(self):
        self.value = self.prior_value
        self.activate()

    def set_constant_context(self):
        self.value = self.context_value
        self.activate()

    def set_context_prior(self, value):
        """
        Value is a boolean tensor of shape (batch_size). True means context, False means prior.
        """
        value = value.to(torch.float32)
        self.value = self.context_value * value + self.prior_value * (1 - value)
        self.activate()

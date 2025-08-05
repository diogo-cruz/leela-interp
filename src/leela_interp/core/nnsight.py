"""NNsight wrapper for Leela Chess Zero model.

This module provides a wrapper around the Leela Chess Zero model to make it
compatible with the nnsight library for neural network introspection and
interpretability analysis.
"""

import torch
from leela_interp.core.lc0 import Lc0Model
from leela_interp.core.leela_board import LeelaBoard
from nnsight import NNsight


class Lc0sight(NNsight):
    """Wrapper around Leela to make it easier to use with `nnsight`.

    Use this instead of the default `NNsight`.

    Note that `Lc0sight.output` will always give you logits, not probabilities!
    (like `Lc0Model.forward()` but unlike `Lc0Model.play()`) To convert these
    logits to probabilities, use `Lc0Model.logits_to_probs()`, not a softmax!
    (since illegal logits need to be filtered out first)

    This inherits all methods from `Lc0Model`, so you can just use it like a normal
    model outside of `nnsight` as well.
    """

    def __init__(self, path=None, device=None):
        """Initialize the Lc0sight model wrapper.
        
        Args:
            path (str, optional): Path to the ONNX model file. If None, uses default.
            device (str, optional): Device to run the model on (e.g., 'cuda', 'cpu').
        """
        model = Lc0Model(onnx_model_path=path, device=device)
        super().__init__(model)
        self._layers = [{} for _ in range(self._model.N_LAYERS)]
        layer_names = [f"encoder{i}" for i in range(self._model.N_LAYERS)]
        for name, _ in model.named_modules():
            if name == "_lc0_model" or name == "":
                continue
            assert name.startswith("_lc0_model.")
            name = name[len("_lc0_model.") :]
            try:
                layer_name, *module_name = name.split("/")
                layer_index = layer_names.index(layer_name)
                module_name = "/".join(module_name)
                self._layers[layer_index][module_name] = getattr(
                    self._envoy._lc0_model, name
                )
            except ValueError:
                # Not a layer module.
                continue

    @property
    def device(self):
        """Get the device the model is running on.
        
        Returns:
            torch.device: The device (CPU/GPU) the model is on.
        """
        return self._model.device

    def trace(self, *args, grads: bool = False, **kwargs):
        """Trace the model execution with nnsight.
        
        Args:
            *args: Variable length argument list passed to parent trace method.
            grads (bool, optional): Whether to compute gradients. Defaults to False.
            **kwargs: Arbitrary keyword arguments passed to parent trace method.
            
        Returns:
            The result of the parent trace method.
            
        Raises:
            ValueError: If grads is passed in invoker_args instead of as direct parameter.
        """
        if "invoker_args" not in kwargs:
            kwargs["invoker_args"] = {}
        if "grads" in kwargs["invoker_args"]:
            raise ValueError("Please pass grad directly to trace()")
        kwargs["invoker_args"]["grads"] = grads
        # HACK: nnsight currently doesn't work with our ONNX model when scanning,
        # so we always need to disable that.
        return super().trace(*args, **kwargs, scan=False, validate=False)

    def _execute(self, *prepared_inputs, **kwargs) -> torch.Tensor:
        """Execute the model with prepared inputs.
        
        Args:
            *prepared_inputs: Prepared input tensors for the model.
            **kwargs: Additional keyword arguments for model execution.
            
        Returns:
            torch.Tensor: Model output logits.
        """
        return self._model(
            *prepared_inputs,
            **kwargs,
        )

    def _prepare_inputs(self, *inputs, grads=False, **kwargs) -> tuple[tuple, int]:
        """Prepare inputs for model execution.
        
        Args:
            *inputs: Input data - either torch.Tensor or LeelaBoard objects.
            grads (bool, optional): Whether to enable gradients. Defaults to False.
            **kwargs: Additional keyword arguments.
            
        Returns:
            tuple[tuple, int]: Tuple containing (prepared_inputs, batch_size).
            
        Raises:
            AssertionError: If number of inputs is not exactly 1.
        """
        assert len(inputs) == 1
        if isinstance(inputs[0], torch.Tensor):
            return inputs, len(inputs[0])

        boards = inputs[0]
        if isinstance(boards, LeelaBoard):
            boards = [boards]
        model_inputs = self._model.make_inputs(boards)
        if grads:
            print("Enabling gradients")
            # This way, everything will have gradients. Would be nicer to only
            # compute the ones we actually need but I don't think nnsight supports that.
            model_inputs.requires_grad = True

        return (model_inputs,), len(model_inputs)

    @property
    def layers(self):
        """Get the layers of the model for introspection.
        
        Returns:
            list: List of dictionaries containing layer modules.
        """
        return self._layers

    def attention_scores(
        self,
        layer: int,
        pre_softmax: bool = False,
        QK_only: bool = False,
        smolgen_only: bool = False,
    ):
        """Get attention scores from a specific layer.
        
        Args:
            layer (int): Layer index to get attention scores from.
            pre_softmax (bool, optional): If True, return pre-softmax weights. Defaults to False.
            QK_only (bool, optional): If True, return only QK scores. Defaults to False.
            smolgen_only (bool, optional): If True, return only smolgen output. Defaults to False.
            
        Returns:
            torch.Tensor: Attention scores based on the specified parameters.
        """
        if pre_softmax:
            return self.layers[layer]["smolgen_weights"]
        elif QK_only:
            return self.layers[layer]["mha/QK/scale"]
        elif smolgen_only:
            return self.layers[layer]["smolgen/out/reshape"]
        return self.layers[layer]["mha/QK/softmax"]

    def residual_stream(self, layer: int, pre_mlp: bool = False):
        """Get the residual stream from a specific layer.
        
        Args:
            layer (int): Layer index to get residual stream from.
            pre_mlp (bool, optional): If True, return pre-MLP residual stream. Defaults to False.
            
        Returns:
            torch.Tensor: Residual stream tensor from the specified layer.
        """
        if pre_mlp:
            return self._lc0_model.post_attention[layer]
        return self._lc0_model.post_mlp[layer]

    def mlp_output(self, layer: int):
        """Get the MLP output from a specific layer.
        
        Args:
            layer (int): Layer index to get MLP output from.
            
        Returns:
            torch.Tensor: MLP output tensor from the specified layer.
        """
        return self._lc0_model.mlp_output[layer]

    def attention_output(self, layer: int):
        """Get the attention output from a specific layer.
        
        Args:
            layer (int): Layer index to get attention output from.
            
        Returns:
            torch.Tensor: Attention output tensor from the specified layer.
        """
        return self._lc0_model.attention_output[layer]

    def headwise_attention_output(self, layer: int):
        """Returns (batch, heads, squares, d_value) shaped outputs.

        These are concatenated and fed through a linear layer to get `attention_output`.
        """
        return self.layers[layer]["mha/QKV/matmul"]

    def V(self, layer: int):
        """Returns the value vectors for the attention layer.

        Shape: (batch, heads, squares, d_value).
        This is multiplied with the QK matrix to get `headwise_attention_output`.
        """
        return self.layers[layer]["mha/V/transpose"]

import torch
from eegatscale.models.bendr import BendrEncoder

# class HookManager:
#     '''
#     HookManager class to register hooks on model layers and store layer outputs.
#     '''
#     def __init__(self):
#         self.layer_outputs = {}
#         self.hooks = {}

#     def hook_fn(self, module, input, output):
#         '''
#         Stores layer outputs in the layer_outputs dictionary
        
#         Args:
#             module: Model layer
#             input: Input tensor to the layer
#             output: Output tensor from the layer
#         '''
#         self.layer_outputs[module] = output

#     def register_hooks(self, model, condition_fn=None):
#         '''
#         Registers hooks on model layers based on a custom condition.

#         Args:
#             model: Model to register hooks on.
#             condition_fn: Optional function to determine whether to register a hook on a given layer.
#                           If None, hooks are registered on all layers.
#         '''
#         for name, layer in model.named_modules():
#             if condition_fn is None or condition_fn(name, layer):
#                 print(f"Registering hook on layer: {name, layer}")
#                 self.hooks[name] = layer.register_forward_hook(self.hook_fn)


class HookManager:
    '''
    HookManager class to register hooks on specific model layers and store their outputs.
    '''
    def __init__(self, layers_of_interest):
        self.layer_outputs = {}  # Stores activations per layer
        self.hooks = {}  # Stores hook references
        self.layers_of_interest = layers_of_interest  # List of layers to hook

    def hook_fn(self, name):
        ''' Returns a hook function that stores layer outputs using layer names as keys. '''
        def fn(module, input, output):
            if name not in self.layer_outputs:
                self.layer_outputs[name] = []  # Allow multiple outputs per layer
            self.layer_outputs[name].append(output)  # Append new output
        return fn

    def register_hooks(self, model):
        '''
        Registers hooks only on specified layers.

        Args:
            model: The model to register hooks on.
        '''
        for name, layer in model.named_modules():
            if name in self.layers_of_interest:  # Check if layer is in list
                if name not in self.hooks:  # Prevent duplicate hooks
                    print(f"Registering hook on layer: {name}")
                    self.hooks[name] = layer.register_forward_hook(self.hook_fn(name))

    def clear_hooks(self):
        '''
        Removes all registered hooks from the model.
        '''
        for name, hook in self.hooks.items():
            hook.remove()
        self.hooks.clear()
        self.layer_outputs.clear()
        print("All hooks cleared.")


# def load_BendrEncoder(model_path):
#     """This function loads a pre-trained BendrEncoder model from a given path."""
#     state_dict = torch.load(model_path, map_location="cpu")
#     encoder_state_dict = {}
#     for key, value in state_dict.items():
#         if key.startswith("encoder."):
#             encoder_state_dict[key] = value

#     encoder = BendrEncoder(in_features=20, encoder_h=512)# , grad_frac=0.1)
#     encoder.load_state_dict(encoder_state_dict)
#     return encoder.eval()
from .nodes import LoadBAGELModel, Prompt, ImageGeneration

NODE_CLASS_MAPPINGS = {
    "LoadBAGELModel": LoadBAGELModel,
    "Prompt": Prompt,
    "ImageGeneration": ImageGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBAGELModel": "Load BAGEL Model",
    "Prompt": "Prompt",
    "ImageGeneration": "Image Generation",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

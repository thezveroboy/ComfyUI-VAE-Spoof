# ComfyUI-VAE-Spoof — декодируй латенты Krea2 декодером Qwen-Image.
#
# VAE у Qwen-Image (1.x), Wan 2.1 и Krea2 — одна архитектура (f8c16) и одно
# латентное пространство. Нода VAE Spoof Loader берёт файл VAE в любом из двух
# форматов ключей (ComfyUI или Diffusers), при нужде переименовывает ключи
# и возвращает обычный объект VAE — дальше работает штатный VAEDecode.

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

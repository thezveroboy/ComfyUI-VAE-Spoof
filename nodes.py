# nodes.py — ноды ComfyUI-VAE-Spoof.
#
# Идея: VAE у Qwen-Image (1.x), Wan 2.1 и Krea2 — одна архитектура (f8c16)
# и одно латентное пространство (те же latents_mean/std; Krea2 вообще грузит
# AutoencoderKLQwenImage из Qwen/Qwen-Image). Поэтому декодер Qwen-Image
# декодирует латенты Krea2 напрямую — нужно только ЗАГРУЗИТЬ файл:
# у официальных репозиториев ключи в формате Diffusers
# (down_blocks/mid_block/up_blocks/quant_conv), а ComfyUI понимает только
# свой формат (downsamples/middle/upsamples/conv1) и сам чинит лишь SD-ключи.
# Нода определяет формат, переименовывает ключи и отдаёт словарь в comfy.sd.VAE.

import folder_paths

from .keymap import (
    MARKER_KEYS,
    check_compat,
    detect_format,
    diffusers_wan_to_comfy,
    inspect_report,
    normalize_keys,
)


def _load_vae_file(vae_name):
    """Прочитать файл VAE из папки models/vae. Возвращает (sd, metadata)."""
    import comfy.utils

    vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
    try:
        loaded = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            return loaded[0], loaded[1], vae_path
    except TypeError:
        pass
    sd = comfy.utils.load_torch_file(vae_path)
    return sd, None, vae_path


def _build_vae(sd, metadata, vae_path):
    """Собрать объект comfy.sd.VAE из словаря (ключи уже в comfy-формате)."""
    import comfy.sd

    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    # Фабрика перезагрузки для мульти-GPU клонов (есть не во всех версиях).
    try:
        if vae_path is not None and hasattr(comfy.sd, "load_vae_patcher"):
            vae.patcher.cached_patcher_init = (
                comfy.sd.load_vae_patcher, (vae_path, metadata, None))
    except Exception:
        pass
    return vae


def _prepare_sd(vae_name):
    """Загрузить файл, определить формат, при нужде сконвертировать.

    Возвращает (sd_comfy, info), где info — dict с полями vae_name, fmt,
    compat_ok, compat_msg, conv_report.
    """
    sd_raw, metadata, vae_path = _load_vae_file(vae_name)
    # Маркерные ключи перепаковщиков (нулевые тензоры) — выкинуть сразу.
    sd_raw = {k: v for k, v in sd_raw.items() if k not in MARKER_KEYS}
    sd = normalize_keys(sd_raw)
    fmt = detect_format(sd.keys())

    conv_report = None
    if fmt == "diffusers_wan":
        sd, conv_report = diffusers_wan_to_comfy(sd)
        fmt_after = detect_format(sd.keys())
        conv_report["format_after"] = fmt_after

    compat_ok, compat_msg = check_compat(sd)
    info = {
        "vae_name": vae_name,
        "fmt": fmt,
        "compat_ok": compat_ok,
        "compat_msg": compat_msg,
        "conv_report": conv_report,
        "metadata": metadata,
        "vae_path": vae_path,
        "raw_keys": list(sd_raw.keys()),
    }
    return sd, info


class VAESpoofLoader:
    """Загрузчик VAE семейства Wan 2.1 / Qwen-Image / Krea2 с авто-ремапом ключей.

    Берёт любой файл из models/vae (comfy- или diffusers-формат) и возвращает
    обычный объект VAE. Дальше — штатный VAEDecode: латенты Krea2 декодируются
    декодером Qwen-Image напрямую, т.к. латентное пространство общее.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
            },
            "optional": {
                "print_report": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae_spoof"
    CATEGORY = "VAE Spoof"

    def load_vae_spoof(self, vae_name, print_report=True):
        sd, info = _prepare_sd(vae_name)
        if not info["compat_ok"]:
            raise ValueError(
                "VAE-Spoof: файл '%s' несовместим с латентами Krea2: %s"
                % (vae_name, info["compat_msg"])
            )
        vae = _build_vae(sd, info["metadata"], info["vae_path"])
        if print_report:
            print(inspect_report(
                info["vae_name"], info["raw_keys"], info["fmt"],
                info["compat_ok"], info["compat_msg"], info["conv_report"]))
        return (vae,)


class VAESpoofInspector:
    """Диагностика файла VAE: формат ключей и совместимость с латентами Krea2.

    Ничего не грузит в модель — только читает имена ключей и формы пары весов.
    Ответом на «ключи не совпадают» будет точный формат и вердикт.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "inspect_vae"
    CATEGORY = "VAE Spoof"
    OUTPUT_NODE = True

    def inspect_vae(self, vae_name):
        sd, info = _prepare_sd(vae_name)
        report = inspect_report(
            info["vae_name"], info["raw_keys"], info["fmt"],
            info["compat_ok"], info["compat_msg"], info["conv_report"])
        print(report)
        return {"ui": {"text": [report]}, "result": (report,)}


NODE_CLASS_MAPPINGS = {
    "VAESpoofLoader": VAESpoofLoader,
    "VAESpoofInspector": VAESpoofInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAESpoofLoader": "VAE Spoof Loader (Qwen/Wan/Krea)",
    "VAESpoofInspector": "VAE Spoof Key Inspector",
}

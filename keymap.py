# keymap.py — ремап ключей VAE семейства Wan 2.1 / Qwen-Image / Krea2.
#
# Два формата имён для ОДНИХ И ТЕ ЖЕ весов:
#   * "comfy"      — формат ComfyUI: encoder.downsamples.*, decoder.middle.*,
#                    decoder.upsamples.*, encoder.conv1 / conv1 / conv2 ...
#   * "diffusers"  — формат Diffusers (официальные репозитории Qwen/Qwen-Image,
#                    Wan-AI, krea-2): encoder.down_blocks.*, mid_block.*,
#                    decoder.up_blocks.*, quant_conv / post_quant_conv ...
#
# ComfyUI сам умеет конвертировать только SD-diffusers ключи, а Wan/Qwen-diffusers
# ключи — нет (отсюда "ключи не совпадают" при загрузке официального VAE).
# Этот модуль делает преобразование diffusers -> comfy, после чего словарь
# можно отдать в comfy.sd.VAE напрямую.
#
# Логика инвертирована из открытого конвертера diffusers (convert_wan_to_diffusers.py,
# направление comfy -> diffusers) и сверена с Wan2GP convert_diffusers_qwen_vae.py.
# Здесь — собственная реализация, без заимствованного кода.
#
# Модуль намеренно не импортирует torch/comfy: значения словаря переносятся
# как есть, для проверки совместимости достаточно форм тензоров.

# Ожидаемая геометрия латентного пространства Qwen-Image 1.x / Wan 2.1 / Krea2.
EXPECTED_Z_DIM = 16        # латентных каналов (f8c16)
EXPECTED_OUT_CHANNELS = 3  # каналов картинки на выходе декодера (RGB)

# Эталонные статистики латентов Qwen-Image VAE (те же у Wan 2.1 и Krea2 —
# Krea2 грузит AutoencoderKLQwenImage из Qwen/Qwen-Image напрямую).
QWEN_LATENTS_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
QWEN_LATENTS_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)

# Префиксы-обёртки, которые срезаются до определения формата
# (чекпоинты целиком, diffusers-папки и т.п.).
STRIP_PREFIXES = ("first_stage_model.", "vae.", "model.")

# Маркерные ключи без весов, которые добавляют перепаковщики (нулевой тензор).
# На загрузку не влияют, из отчёта исключаются.
MARKER_KEYS = ("qwen_image",)


def strip_prefix(key):
    """Срезать известный префикс-обёртку, если есть."""
    for pref in STRIP_PREFIXES:
        if key.startswith(pref):
            return key[len(pref):]
    return key


def normalize_keys(state_dict):
    """Вернуть словарь со срезанными префиксами (значения не трогает)."""
    return {strip_prefix(k): v for k, v in state_dict.items()}


def detect_format(keys):
    """Определить формат словаря VAE по именам ключей.

    Возвращает одно из:
      'comfy_wan'      — ComfyUI-формат Wan/Qwen (downsamples/middle/upsamples)
      'diffusers_wan'  — Diffusers-формат Wan/Qwen (down_blocks/mid_block/up_blocks)
      'sd_diffusers'   — Diffusers-формат обычного SD VAE (сам разберётся ComfyUI)
      'sd_comfy'       — ComfyUI-формат обычного SD VAE
      'unknown'        — ни то ни другое
    """
    key_set = set(keys)
    has = key_set.__contains__

    def any_prefix(prefix):
        return any(k.startswith(prefix) for k in key_set)

    # Diffusers-формат Wan/Qwen: маркеры mid_block / down_blocks / quant_conv.
    # (Голый префикс decoder.up_blocks. здесь НЕ проверяем — он общий
    # с обычным SD VAE; Wan выдаёт связка mid_block + quant_conv + down_blocks.)
    if (has("encoder.mid_block.resnets.0.norm1.gamma")
            or has("decoder.mid_block.resnets.0.norm1.gamma")
            or has("quant_conv.weight")
            or has("post_quant_conv.weight")
            or any_prefix("encoder.mid_block.")
            or any_prefix("decoder.mid_block.")
            or any_prefix("encoder.down_blocks.")):
        return "diffusers_wan"
    # ComfyUI-формат Wan/Qwen.
    if (has("encoder.middle.0.residual.0.gamma")
            or has("decoder.middle.0.residual.0.gamma")
            or has("encoder.conv1.weight")
            or any_prefix("encoder.downsamples.")
            or any_prefix("decoder.upsamples.")):
        return "comfy_wan"
    # Обычный SD VAE в diffusers-формате — ComfyUI конвертирует сам.
    if has("decoder.up_blocks.0.resnets.0.norm1.weight"):
        return "sd_diffusers"
    # Обычный SD VAE в comfy-формате.
    if has("decoder.conv_in.weight") and has("encoder.down.0.block.0.norm1.weight"):
        return "sd_comfy"
    return "unknown"


# --- Точные (неиндексированные) переименования diffusers -> comfy ------------

# Мост квантования латентов.
_QUANT_EXACT = {
    "quant_conv.weight": "conv1.weight",
    "quant_conv.bias": "conv1.bias",
    "post_quant_conv.weight": "conv2.weight",
    "post_quant_conv.bias": "conv2.bias",
}

# Входные свёртки энкодера/декодера.
_CONVIN_EXACT = {
    "encoder.conv_in.weight": "encoder.conv1.weight",
    "encoder.conv_in.bias": "encoder.conv1.bias",
    "decoder.conv_in.weight": "decoder.conv1.weight",
    "decoder.conv_in.bias": "decoder.conv1.bias",
}

# Выходные головы энкодера/декодера.
_HEAD_EXACT = {
    "encoder.norm_out.gamma": "encoder.head.0.gamma",
    "encoder.conv_out.weight": "encoder.head.2.weight",
    "encoder.conv_out.bias": "encoder.head.2.bias",
    "decoder.norm_out.gamma": "decoder.head.0.gamma",
    "decoder.conv_out.weight": "decoder.head.2.weight",
    "decoder.conv_out.bias": "decoder.head.2.bias",
}

# Средние блоки: mid_block.resnets.{0,1} -> middle.{0,2}, attentions.0 -> middle.1.
_MID_RULES = (
    ("mid_block.resnets.0.norm1.gamma", "middle.0.residual.0.gamma"),
    ("mid_block.resnets.0.conv1.", "middle.0.residual.2."),
    ("mid_block.resnets.0.norm2.gamma", "middle.0.residual.3.gamma"),
    ("mid_block.resnets.0.conv2.", "middle.0.residual.6."),
    ("mid_block.attentions.0.norm.gamma", "middle.1.norm.gamma"),
    ("mid_block.attentions.0.to_qkv.", "middle.1.to_qkv."),
    ("mid_block.attentions.0.proj.", "middle.1.proj."),
    ("mid_block.resnets.1.norm1.gamma", "middle.2.residual.0.gamma"),
    ("mid_block.resnets.1.conv1.", "middle.2.residual.2."),
    ("mid_block.resnets.1.norm2.gamma", "middle.2.residual.3.gamma"),
    ("mid_block.resnets.1.conv2.", "middle.2.residual.6."),
)

# Переименования внутри residual-блоков (действуют в обе стороны таблицы).
_RES_RULES = (
    (".norm1.gamma", ".residual.0.gamma"),
    (".conv1.", ".residual.2."),
    (".norm2.gamma", ".residual.3.gamma"),
    (".conv2.", ".residual.6."),
    (".conv_shortcut.", ".shortcut."),
)

# Плоские индексы апсемплеров декодера в comfy-формате по номеру up-блока.
# diffusers: decoder.up_blocks.{0,1,2}.upsamplers.0
# comfy:     decoder.upsamples.{3,7,11}
_UPSAMPLER_FLAT = {"0": "3", "1": "7", "2": "11"}


def _convert_down_key(key):
    """encoder.down_blocks.* (diffusers) -> encoder.downsamples.* (comfy)."""
    new_key = key.replace("encoder.down_blocks.", "encoder.downsamples.", 1)
    for diff_part, comfy_part in _RES_RULES:
        new_key = new_key.replace(diff_part, comfy_part)
    return new_key


def _convert_up_key(key):
    """decoder.up_blocks.* (diffusers) -> decoder.upsamples.* (comfy).

    Возвращает новый ключ или None, если правило не подошло.
    """
    parts = key.split(".")
    # Residual-блоки: decoder.up_blocks.{B}.resnets.{R}.* -> плоский индекс B*4+R.
    if (len(parts) >= 6 and parts[2].isdigit()
            and parts[3] == "resnets" and parts[4].isdigit()):
        flat = int(parts[2]) * 4 + int(parts[4])
        new_key = ".".join(["decoder", "upsamples", str(flat)] + parts[5:])
        for diff_part, comfy_part in _RES_RULES:
            new_key = new_key.replace(diff_part, comfy_part)
        return new_key
    # Апсемплеры и временные свёртки: up_blocks.{0,1,2}.upsamplers.0 -> upsamples.{3,7,11}.
    if ".upsamplers.0." in key and len(parts) >= 3 and parts[2] in _UPSAMPLER_FLAT:
        flat = _UPSAMPLER_FLAT[parts[2]]
        return key.replace(
            "decoder.up_blocks.%s.upsamplers.0" % parts[2],
            "decoder.upsamples.%s" % flat,
            1,
        )
    return None


def diffusers_wan_to_comfy(state_dict):
    """Перевести словарь Wan/Qwen VAE из формата Diffusers в формат ComfyUI.

    На вход — словарь {имя: тензор}, на выход — (новый_словарь, отчёт),
    где отчёт — dict {converted, passthrough, unmapped}.
    Входной словарь не меняется.
    """
    out = {}
    converted = 0
    passthrough = 0
    unmapped = []

    for key, value in state_dict.items():
        # 1) Точные переименования.
        if key in _QUANT_EXACT:
            out[_QUANT_EXACT[key]] = value
            converted += 1
            continue
        if key in _CONVIN_EXACT:
            out[_CONVIN_EXACT[key]] = value
            converted += 1
            continue
        if key in _HEAD_EXACT:
            out[_HEAD_EXACT[key]] = value
            converted += 1
            continue

        # 2) Средние блоки (mid_block -> middle), отдельно энкодер/декодер.
        mid_done = False
        for side in ("encoder", "decoder"):
            for diff_part, comfy_part in _MID_RULES:
                old = side + "." + diff_part
                if key == old or key.startswith(old):
                    out[key.replace(old, side + "." + comfy_part, 1)] = value
                    converted += 1
                    mid_done = True
                    break
            if mid_done:
                break
        if mid_done:
            continue

        # 3) Нисходящий тракт энкодера.
        if key.startswith("encoder.down_blocks."):
            out[_convert_down_key(key)] = value
            converted += 1
            continue

        # 4) Восходящий тракт декодера.
        if key.startswith("decoder.up_blocks."):
            new_key = _convert_up_key(key)
            if new_key is not None:
                out[new_key] = value
                converted += 1
                continue
            unmapped.append(key)
            out[key] = value
            continue

        # 5) Всё остальное — как есть (метаданные, хвосты новых версий).
        passthrough += 1
        out[key] = value

    report = {
        "converted": converted,
        "passthrough": passthrough,
        "unmapped": unmapped,
    }
    return out, report


def _shape_of(value):
    """Достать форму значения без зависимости от torch (shape или size())."""
    shape = getattr(value, "shape", None)
    if shape is None and hasattr(value, "size"):
        try:
            shape = tuple(value.size())
        except Exception:
            shape = None
    if shape is not None:
        try:
            return tuple(int(x) for x in shape)
        except Exception:
            return None
    return None


def check_compat(state_dict):
    """Проверить, что словарь — f8c16 VAE (Qwen-Image 1.x / Wan 2.1 / Krea2).

    state_dict — уже нормализованный (comfy-имена).
    Возвращает (ok: bool, message: str).
    """
    quant_key = None
    for cand in ("conv1.weight", "quant_conv.weight"):
        if cand in state_dict:
            quant_key = cand
            break
    if quant_key is None:
        return False, "не найден мост квантования (conv1/quant_conv) — не VAE Wan-семейства"

    shape = _shape_of(state_dict[quant_key])
    if shape is None or len(shape) < 2:
        return False, "не удалось прочитать форму %s" % quant_key
    z_dim = shape[0] // 2
    if z_dim != EXPECTED_Z_DIM:
        return False, (
            "латентных каналов: %d, ожидалось %d. "
            "Это НЕ VAE Qwen-Image 1.x / Wan 2.1 / Krea2 "
            "(похоже на Qwen-Image 2.x f16c64 — им латенты Krea2 не декодировать)"
            % (z_dim, EXPECTED_Z_DIM)
        )

    head_key = None
    for cand in ("decoder.head.2.weight", "decoder.conv_out.weight"):
        if cand in state_dict:
            head_key = cand
            break
    if head_key is not None:
        out_shape = _shape_of(state_dict[head_key])
        if out_shape is not None and out_shape[0] != EXPECTED_OUT_CHANNELS:
            return False, (
                "каналов на выходе декодера: %d, ожидалось %d (RGB). "
                "RGBA-VAE Qwen-Image 2.x с латентами Krea2 несовместим"
                % (out_shape[0], EXPECTED_OUT_CHANNELS)
            )

    return True, "f8c16, %d латентных каналов — совместим с латентами Krea2" % z_dim


def inspect_report(vae_name, raw_keys, fmt, compat_ok, compat_msg, conv_report=None):
    """Собрать человекочитаемый отчёт для ноды-инспектора."""
    lines = []
    lines.append("Файл: %s" % vae_name)
    lines.append("Ключей всего: %d" % len(raw_keys))
    fmt_names = {
        "comfy_wan": "ComfyUI-формат Wan/Qwen (downsamples/middle/upsamples)",
        "diffusers_wan": "Diffusers-формат Wan/Qwen (down_blocks/mid_block/up_blocks)",
        "sd_diffusers": "Diffusers-формат обычного SD VAE",
        "sd_comfy": "ComfyUI-формат обычного SD VAE",
        "unknown": "неизвестный формат",
    }
    lines.append("Формат: %s" % fmt_names.get(fmt, fmt))
    if conv_report is not None:
        lines.append("Переименовано ключей: %d" % conv_report.get("converted", 0))
        lines.append("Оставлено как есть: %d" % conv_report.get("passthrough", 0))
        unmapped = conv_report.get("unmapped", [])
        if unmapped:
            lines.append("Без правила (%d), первые:" % len(unmapped))
            for k in unmapped[:10]:
                lines.append("  - %s" % k)
    lines.append("Совместимость с Krea2: %s — %s"
                 % ("ДА" if compat_ok else "НЕТ", compat_msg))
    if fmt == "diffusers_wan":
        lines.append("Штатный VAELoader такой файл не возьмёт — грузите через VAE Spoof Loader.")
    elif fmt == "comfy_wan":
        lines.append("Штатный VAELoader такой файл берёт и сам — спoof не обязателен.")
    # Парочка примеров ключей для глаз.
    sample = [k for k in raw_keys if k not in MARKER_KEYS][:6]
    if sample:
        lines.append("Примеры ключей:")
        for k in sample:
            lines.append("  - %s" % k)
    return "\n".join(lines)

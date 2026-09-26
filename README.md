# ComfyUI-VAE-Spoof

Декодируй латенты **Krea2** декодером **Qwen-Image** (или Wan 2.1). Одна нода-загрузчик с авто-ремапом ключей.

## Почему это работает

Перепроверено по исходникам и техрепортам:

1. **Qwen-Image (1.x) VAE построен на архитектуре Wan 2.1 VAE** — общий энкодер заморожен, дообучен только image-декодер (Qwen-Image Technical Report, §2.3). Латент: f8, 16 каналов.
2. **Krea2 использует именно Qwen-Image VAE** — в их репозитории `autoencoder.py` грузится `AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae")`. Латентное пространство и статистики (`latents_mean/std`) — те же самые.
3. Итог: декодер Qwen-Image декодирует латенты Krea2 **напрямую**, без адаптации весов. Проблема только в **загрузке файла**: у официальных репозиториев ключи в формате Diffusers (`down_blocks/mid_block/up_blocks/quant_conv`), а ComfyUI понимает свой формат (`downsamples/middle/upsamples/conv1`) и сам чинит только SD-ключи. Отсюда «ключи не совпадают».

## Ноды

| Нода | Что делает |
|---|---|
| **VAE Spoof Loader (Qwen/Wan/Krea)** | Грузит любой VAE семейства из `models/vae` (comfy- или diffusers-формат), при нужде переименовывает ключи, возвращает обычный `VAE`. Дальше — штатный `VAEDecode`. |
| **VAE Spoof Key Inspector** | Диагностика файла: формат ключей + вердикт совместимости с латентами Krea2. Ничего не грузит в модель. |

## Использование

1. Положи файл VAE в `ComfyUI/models/vae/` (например `qwen_image_vae.safetensors`, официальный `diffusion_pytorch_model.safetensors` из `Qwen/Qwen-Image` — тоже подойдёт, переименуй в `*.safetensors`).
2. Поставь **VAE Spoof Loader**, выбери файл.
3. Соедини выход `vae` со штатным **VAEDecode**, на вход латентов подай латенты Krea2. Готово.

Не знаешь, что за файл? Сначала прогони его через **VAE Spoof Key Inspector**.

## ⚠️ Важно: Qwen-Image 2.x НЕ подойдёт

Настоящий VAE из `Qwen/Qwen-Image-2.1` — это **f16c64 RGBA** (16x сжатие, 64 канала + альфа-канал). Им латенты Krea2 (f8c16) не декодировать **никаким** ремапом — разная размерность пространства. Лоадер определяет это по формам весов и отказывается с понятной ошибкой вместо тихого мусора. Если твой файл «qwen image 2.1 vae» — это f8c16 (VAE от Qwen-Image 1.x), всё сходится.

## Установка

```
cd ComfyUI/custom_nodes
git clone https://github.com/thezveroboy/ComfyUI-VAE-Spoof
```
Перезапусти ComfyUI. Зависимостей нет — только `torch`/`safetensors`/`comfy`/`folder_paths` из самого ComfyUI.

## Проверка без ComfyUI

Ремап ключей тестируется чистым питоном (без torch):

```
python tests/test_keymap.py
```

## Источники

- Qwen-Image Technical Report (§2.3 VAE: архитектура Wan-2.1-VAE, замороженный энкодер)
- `krea-ai/krea-2` `autoencoder.py` (QwenAutoencoder = AutoencoderKLQwenImage из Qwen/Qwen-Image)
- `huggingface/diffusers` `scripts/convert_wan_to_diffusers.py` (таблица переименований comfy→diffusers)
- `deepbeepmeep/Wan2GP` `models/qwen/convert_diffusers_qwen_vae.py` (обратное направление)

## Лицензия

MIT

# Тест ремапа ключей без torch/comfy: значения — заглушки, важны только имена.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keymap import (
    check_compat,
    detect_format,
    diffusers_wan_to_comfy,
    normalize_keys,
)


class FakeTensor:
    def __init__(self, *shape):
        self.shape = tuple(shape)


def test_detect():
    comfy_keys = [
        "encoder.conv1.weight",
        "encoder.middle.0.residual.0.gamma",
        "encoder.downsamples.0.residual.0.gamma",
        "decoder.middle.1.norm.gamma",
        "decoder.upsamples.3.resample.weight",
        "decoder.head.2.weight",
        "conv1.weight",
    ]
    assert detect_format(comfy_keys) == "comfy_wan", "comfy_wan не распознан"

    diff_keys = [
        "quant_conv.weight",
        "encoder.conv_in.weight",
        "encoder.mid_block.resnets.0.norm1.gamma",
        "encoder.down_blocks.0.norm1.gamma",
        "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.up_blocks.1.resnets.2.conv1.weight",
        "decoder.norm_out.gamma",
    ]
    assert detect_format(diff_keys) == "diffusers_wan", "diffusers_wan не распознан"

    sd_keys = ["decoder.up_blocks.0.resnets.0.norm1.weight", "encoder.down.0.block.0.norm1.weight"]
    assert detect_format(sd_keys) == "sd_diffusers", "sd_diffusers не распознан"

    assert detect_format(["foo", "bar"]) == "unknown", "unknown не распознан"
    print("detect: OK")


def test_exact_remaps():
    sd = {
        "quant_conv.weight": 1,
        "quant_conv.bias": 2,
        "post_quant_conv.weight": 3,
        "post_quant_conv.bias": 4,
        "encoder.conv_in.weight": 5,
        "encoder.conv_in.bias": 6,
        "decoder.conv_in.weight": 7,
        "decoder.conv_in.bias": 8,
        "encoder.norm_out.gamma": 9,
        "encoder.conv_out.weight": 10,
        "encoder.conv_out.bias": 11,
        "decoder.norm_out.gamma": 12,
        "decoder.conv_out.weight": 13,
        "decoder.conv_out.bias": 14,
    }
    out, rep = diffusers_wan_to_comfy(sd)
    assert out["conv1.weight"] == 1
    assert out["conv1.bias"] == 2
    assert out["conv2.weight"] == 3
    assert out["conv2.bias"] == 4
    assert out["encoder.conv1.weight"] == 5
    assert out["decoder.conv1.bias"] == 8
    assert out["encoder.head.0.gamma"] == 9
    assert out["encoder.head.2.weight"] == 10
    assert out["decoder.head.2.bias"] == 14
    assert rep["converted"] == 14 and rep["passthrough"] == 0
    print("exact: OK")


def test_mid_blocks():
    sd = {
        "encoder.mid_block.resnets.0.norm1.gamma": 1,
        "encoder.mid_block.resnets.0.conv1.weight": 2,
        "encoder.mid_block.resnets.0.norm2.gamma": 3,
        "encoder.mid_block.resnets.0.conv2.bias": 4,
        "encoder.mid_block.attentions.0.to_qkv.weight": 5,
        "encoder.mid_block.attentions.0.proj.bias": 6,
        "encoder.mid_block.resnets.1.norm1.gamma": 7,
        "encoder.mid_block.resnets.1.conv2.weight": 8,
        "decoder.mid_block.resnets.1.norm2.gamma": 9,
        "decoder.mid_block.attentions.0.norm.gamma": 10,
    }
    out, _ = diffusers_wan_to_comfy(sd)
    assert out["encoder.middle.0.residual.0.gamma"] == 1
    assert out["encoder.middle.0.residual.2.weight"] == 2
    assert out["encoder.middle.0.residual.3.gamma"] == 3
    assert out["encoder.middle.0.residual.6.bias"] == 4
    assert out["encoder.middle.1.to_qkv.weight"] == 5
    assert out["encoder.middle.1.proj.bias"] == 6
    assert out["encoder.middle.2.residual.0.gamma"] == 7
    assert out["encoder.middle.2.residual.6.weight"] == 8
    assert out["decoder.middle.2.residual.3.gamma"] == 9
    assert out["decoder.middle.1.norm.gamma"] == 10
    print("mid: OK")


def test_down_up_blocks():
    sd = {
        "encoder.down_blocks.2.norm1.gamma": 1,
        "encoder.down_blocks.2.conv1.weight": 2,
        "encoder.down_blocks.2.norm2.gamma": 3,
        "encoder.down_blocks.2.conv2.bias": 4,
        "encoder.down_blocks.2.conv_shortcut.weight": 5,
        # decoder: up_block 1, resnet 2 -> плоский 1*4+2 = 6
        "decoder.up_blocks.1.resnets.2.norm1.gamma": 6,
        "decoder.up_blocks.1.resnets.2.conv1.weight": 7,
        "decoder.up_blocks.1.resnets.0.conv_shortcut.bias": 8,
        # апсемплеры: up_blocks.0/1/2 -> upsamples.3/7/11
        "decoder.up_blocks.0.upsamplers.0.resample.weight": 9,
        "decoder.up_blocks.1.upsamplers.0.time_conv.weight": 10,
        "decoder.up_blocks.2.upsamplers.0.resample.bias": 11,
    }
    out, _ = diffusers_wan_to_comfy(sd)
    assert out["encoder.downsamples.2.residual.0.gamma"] == 1
    assert out["encoder.downsamples.2.residual.2.weight"] == 2
    assert out["encoder.downsamples.2.residual.3.gamma"] == 3
    assert out["encoder.downsamples.2.residual.6.bias"] == 4
    assert out["encoder.downsamples.2.shortcut.weight"] == 5
    assert out["decoder.upsamples.6.residual.0.gamma"] == 6
    assert out["decoder.upsamples.6.residual.2.weight"] == 7
    assert out["decoder.upsamples.4.shortcut.bias"] == 8
    assert out["decoder.upsamples.3.resample.weight"] == 9
    assert out["decoder.upsamples.7.time_conv.weight"] == 10
    assert out["decoder.upsamples.11.resample.bias"] == 11
    print("down/up: OK")


def test_prefixes_and_compat():
    sd = {
        "first_stage_model.encoder.conv1.weight": FakeTensor(3, 3),
        "first_stage_model.conv1.weight": FakeTensor(32, 32, 1, 1, 1),
        "first_stage_model.decoder.head.2.weight": FakeTensor(3, 96, 3, 3, 3),
    }
    norm = normalize_keys(sd)
    assert "encoder.conv1.weight" in norm
    assert detect_format(norm.keys()) == "comfy_wan"
    ok, msg = check_compat(norm)
    assert ok, msg

    bad = dict(norm)
    bad["conv1.weight"] = FakeTensor(128, 128, 1, 1, 1)  # z=64: Qwen-Image 2.x
    ok, msg = check_compat(bad)
    assert not ok and "2.x" in msg, msg

    rgba = dict(norm)
    rgba["decoder.head.2.weight"] = FakeTensor(4, 96, 3, 3, 3)
    ok, msg = check_compat(rgba)
    assert not ok and "RGBA" in msg, msg
    print("prefix/compat: OK")


if __name__ == "__main__":
    test_detect()
    test_exact_remaps()
    test_mid_blocks()
    test_down_up_blocks()
    test_prefixes_and_compat()
    print("ВСЕ ТЕСТЫ KEYMAP: OK")

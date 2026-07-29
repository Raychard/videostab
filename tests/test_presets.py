"""产品档位: 三档必须真正拉开权衡, 且默认档不能被悄悄改动.

档位参数来自 NUS 144 段全量实测, 见 config.PRESETS 的注释.
"""
import pytest

from videostab.config import DEFAULT_PRESET, PRESETS, PipelineConfig, preset


def test_default_is_recommended():
    """日常默认必须是推荐档 —— 这是产品承诺, 改动需同步文档与评测."""
    assert DEFAULT_PRESET == "recommended"
    assert preset().smoothing.crop_ratio == PRESETS["recommended"]["crop_ratio"]


def test_default_preset_matches_bare_config():
    """推荐档即出厂值: 不传档位名的 PipelineConfig 应与推荐档一致.

    两者一旦分叉, 仓库里既有的全部评测数字就不再对应默认行为.
    """
    bare, rec = PipelineConfig().smoothing, preset("recommended").smoothing
    assert bare.crop_ratio == rec.crop_ratio
    assert bare.anisotropy_cap_ratio == rec.anisotropy_cap_ratio


def test_presets_are_ordered_by_crop():
    """裁切预算严格递增 —— 三档的定义主轴就是它."""
    crops = [PRESETS[n]["crop_ratio"]
             for n in ("minimal_crop", "recommended", "most_stable")]
    assert crops == sorted(crops)
    assert len(set(crops)) == 3          # 必须真的拉开, 不能重合


def test_large_budget_preset_tightens_tau():
    """最稳定档的 τ 必须比推荐档更紧.

    crop=0.16 在 τ=5px 下归一化 distortion 仅 0.8978, 显著劣于推荐档的
    0.9083 (符号检验 z=-2.33); 收紧到 4px 才追平. 若有人把它调回 5px,
    该档就成了"更糊且更费画幅", 失去存在意义.
    """
    assert (PRESETS["most_stable"]["anisotropy_cap_ratio"]
            < PRESETS["recommended"]["anisotropy_cap_ratio"])


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_builds_and_caps_anisotropy(name):
    cfg = preset(name)
    assert 0 < cfg.smoothing.crop_ratio < 0.3      # 0.30 实测全面崩坏
    assert cfg.smoothing.anisotropy_cap_ratio > 0  # 各档均须开启封顶


def test_overrides_reach_pipeline_fields():
    cfg = preset("most_stable", device="cuda", proxy_height=360)
    assert (cfg.device, cfg.proxy_height) == ("cuda", 360)
    assert cfg.smoothing.crop_ratio == 0.16        # 档位值不被覆盖冲掉


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="未知档位"):
        preset("ultra_stable")

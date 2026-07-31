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
    assert bare.corner_space == rec.corner_space


def test_presets_are_ordered_by_crop():
    """裁切预算严格递增 —— 三档的定义主轴就是它."""
    crops = [PRESETS[n]["crop_ratio"]
             for n in ("minimal_crop", "recommended", "most_stable")]
    assert crops == sorted(crops)
    assert len(set(crops)) == 3          # 必须真的拉开, 不能重合


def test_all_presets_jello_immune():
    """三档必须全部运行在角点空间 —— "极度厌恶果冻"是产品要求.

    角点空间下输出场恒为单应位移, 直线弯曲结构性为零(NUS 实测弯曲 p95
    从 5.6px 降到 0.13px). 谁把某档切回顶点空间, 果冻就回来了.
    """
    for name in PRESETS:
        assert PRESETS[name].get("corner_space") is True
        assert preset(name).smoothing.corner_space is True


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_builds_and_caps_anisotropy(name):
    cfg = preset(name)
    assert 0 < cfg.smoothing.crop_ratio < 0.3      # 0.30 实测全面崩坏


def test_overrides_reach_pipeline_fields():
    cfg = preset("most_stable", device="cuda", proxy_height=360)
    assert (cfg.device, cfg.proxy_height) == ("cuda", 360)
    assert cfg.smoothing.crop_ratio == 0.16        # 档位值不被覆盖冲掉


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="未知档位"):
        preset("ultra_stable")

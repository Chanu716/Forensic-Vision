import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class Config:
    values: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def validate_config(config: Config) -> list[str]:
    """Inspects configuration for inert, deprecated, or misleading options and returns warnings."""
    warnings_list: list[str] = []
    model_cfg = config.values.get("model", {})
    loc_cfg = config.values.get("localization", {})
    arch = str(model_cfg.get("arch", "")).lower()

    # 1. use_gradient_guidance: inert on dual-stream models
    if "dual_stream" in arch and model_cfg.get("use_gradient_guidance", False):
        warnings_list.append(
            f"Configuration 'model.use_gradient_guidance: true' is inert for architecture '{arch}'. "
            "Dual-stream models directly compute consecutive frame differences (|K_f - K_{f+1}|) "
            "without Sobel spatial-gradient guidance."
        )

    # 2. use_pretrained: inert because no pretrained weights are loaded
    if model_cfg.get("use_pretrained", False):
        warnings_list.append(
            "Configuration 'model.use_pretrained: true' is inert. "
            "No pretrained weight checkpoint is bundled or downloaded; "
            "the model backbone is initialized with random weights."
        )

    # 3. adaptive_sensitivity: inert in median-drop formula
    if loc_cfg.get("adaptive_sensitivity") is not None and loc_cfg.get("use_adaptive_threshold", True):
        warnings_list.append(
            f"Configuration 'localization.adaptive_sensitivity: {loc_cfg.get('adaptive_sensitivity')}' is inert. "
            "The current adaptive thresholding function computes dynamic threshold as "
            "max(0.35, min(threshold, median - 0.14)); adaptive_sensitivity is unused."
        )

    # 4. use_multiscale: note single-scale mode
    if loc_cfg.get("use_multiscale") is False:
        warnings_list.append(
            "Configuration notice: 'localization.use_multiscale: false'. "
            "Temporal localization evaluates single-scale SSIM rather than Multi-Scale SSIM (MS-SSIM)."
        )

    return warnings_list


def load_config(path: str | Path, warn: bool = True) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    cfg = Config(values=values)
    if warn:
        detected_warnings = validate_config(cfg)
        for w in detected_warnings:
            warnings.warn(w, UserWarning, stacklevel=2)
    return cfg


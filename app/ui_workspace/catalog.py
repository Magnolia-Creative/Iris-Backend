from __future__ import annotations

from dataclasses import dataclass

from app.intent_compiler.capabilities import DEFAULT_EFFECT_CAPABILITIES
from app.ui_workspace.models import ControlType, WidgetRole


CATALOG_VERSION = "1"


@dataclass(frozen=True)
class CatalogParameter:
    id: str
    label: str
    control: ControlType
    min_value: float
    max_value: float
    default_value: float
    effect_operation: str | None = None
    effect_param_name: str | None = None


@dataclass(frozen=True)
class CatalogWidget:
    id: str
    display_name: str
    role: WidgetRole
    supported_modalities: frozenset[str]
    intent_contributions: frozenset[str]
    variants: frozenset[str]
    valid_controls: frozenset[ControlType]


WIDGET_CATALOG: dict[str, CatalogWidget] = {
    "playback.viewer": CatalogWidget(
        id="playback.viewer",
        display_name="Playback",
        role=WidgetRole.monitor,
        supported_modalities=frozenset({"visual", "audio", "mixed"}),
        intent_contributions=frozenset({"preview", "scrub"}),
        variants=frozenset({"compact", "large"}),
        valid_controls=frozenset(),
    ),
    "playback.beforeAfterViewer": CatalogWidget(
        id="playback.beforeAfterViewer",
        display_name="Before/After Preview",
        role=WidgetRole.monitor,
        supported_modalities=frozenset({"visual"}),
        intent_contributions=frozenset({"compare_before_after", "preview"}),
        variants=frozenset({"large", "split"}),
        valid_controls=frozenset(),
    ),
    "timeline.full": CatalogWidget(
        id="timeline.full",
        display_name="Full Timeline",
        role=WidgetRole.representation,
        supported_modalities=frozenset({"timeline", "mixed"}),
        intent_contributions=frozenset({"navigate_timeline", "edit_structure"}),
        variants=frozenset({"expanded", "compressed"}),
        valid_controls=frozenset(),
    ),
    "timeline.primaryTrack": CatalogWidget(
        id="timeline.primaryTrack",
        display_name="Primary Video Track",
        role=WidgetRole.representation,
        supported_modalities=frozenset({"visual", "timeline"}),
        intent_contributions=frozenset({"focus_clip", "trim_range"}),
        variants=frozenset({"affectedRangeOnly", "full"}),
        valid_controls=frozenset(),
    ),
    "timeline.focusedClipStrip": CatalogWidget(
        id="timeline.focusedClipStrip",
        display_name="Focused Clip Strip",
        role=WidgetRole.representation,
        supported_modalities=frozenset({"visual", "timeline"}),
        intent_contributions=frozenset({"focus_clip", "scrub"}),
        variants=frozenset({"compact", "expanded"}),
        valid_controls=frozenset(),
    ),
    "audio.levelsMeter": CatalogWidget(
        id="audio.levelsMeter",
        display_name="Audio Levels",
        role=WidgetRole.monitor,
        supported_modalities=frozenset({"audio"}),
        intent_contributions=frozenset({"monitor_audio"}),
        variants=frozenset({"compact", "expanded"}),
        valid_controls=frozenset(),
    ),
    "toolbar.parameterControls": CatalogWidget(
        id="toolbar.parameterControls",
        display_name="Parameter Controls",
        role=WidgetRole.inspector,
        supported_modalities=frozenset({"visual", "audio", "mixed"}),
        intent_contributions=frozenset({"tune_parameter"}),
        variants=frozenset({"sliderGroup", "compact"}),
        valid_controls=frozenset({ControlType.slider, ControlType.toggle, ControlType.segmented}),
    ),
    "toolbar.reviewActions": CatalogWidget(
        id="toolbar.reviewActions",
        display_name="Review Actions",
        role=WidgetRole.review,
        supported_modalities=frozenset({"mixed"}),
        intent_contributions=frozenset({"approve_edit", "reject_edit", "refine_prompt"}),
        variants=frozenset({"standard"}),
        valid_controls=frozenset({ControlType.button}),
    ),
    "toolbar.promptBar": CatalogWidget(
        id="toolbar.promptBar",
        display_name="Prompt Bar",
        role=WidgetRole.tool,
        supported_modalities=frozenset({"mixed"}),
        intent_contributions=frozenset({"enter_prompt", "refine_prompt"}),
        variants=frozenset({"idle", "typing", "recording"}),
        valid_controls=frozenset(),
    ),
    "toolbar.clipTools": CatalogWidget(
        id="toolbar.clipTools",
        display_name="Clip Tools",
        role=WidgetRole.tool,
        supported_modalities=frozenset({"timeline"}),
        intent_contributions=frozenset({"split_clip", "delete_clip", "color_clip", "volume_clip"}),
        variants=frozenset({"collapsed", "expanded"}),
        valid_controls=frozenset(),
    ),
    "panel.importBrowser": CatalogWidget(
        id="panel.importBrowser",
        display_name="Import Browser",
        role=WidgetRole.navigator,
        supported_modalities=frozenset({"import"}),
        intent_contributions=frozenset({"browse_media", "import_media"}),
        variants=frozenset({"standard"}),
        valid_controls=frozenset(),
    ),
    "panel.exportSettings": CatalogWidget(
        id="panel.exportSettings",
        display_name="Export Settings",
        role=WidgetRole.tool,
        supported_modalities=frozenset({"export"}),
        intent_contributions=frozenset({"configure_export"}),
        variants=frozenset({"standard"}),
        valid_controls=frozenset(),
    ),
}


def _build_effect_parameters() -> dict[str, CatalogParameter]:
    parameters: dict[str, CatalogParameter] = {}
    operation_labels = {
        "addGrain": ("grain", "Grain"),
        "setTemperature": ("temperature", "Temperature"),
        "setSaturation": ("saturation", "Saturation"),
        "setContrast": ("contrast", "Contrast"),
        "setExposure": ("exposure", "Exposure"),
        "setHighlights": ("highlights", "Highlights"),
        "setShadows": ("shadows", "Shadows"),
    }
    for capability in DEFAULT_EFFECT_CAPABILITIES:
        mapping = operation_labels.get(capability.operation)
        if mapping is None:
            continue
        param_id, label = mapping
        param_spec = capability.parameters[0] if capability.parameters else None
        if param_spec is None:
            continue
        minimum = float(param_spec.minimum if param_spec.minimum is not None else -1)
        maximum = float(param_spec.maximum if param_spec.maximum is not None else 1)
        parameters[param_id] = CatalogParameter(
            id=param_id,
            label=label,
            control=ControlType.slider,
            min_value=minimum,
            max_value=maximum,
            default_value=0.0,
            effect_operation=capability.operation,
            effect_param_name=param_spec.name,
        )
    parameters["vintageIntensity"] = CatalogParameter(
        id="vintageIntensity",
        label="Vintage",
        control=ControlType.slider,
        min_value=0.0,
        max_value=1.0,
        default_value=0.7,
    )
    parameters["volumeGain"] = CatalogParameter(
        id="volumeGain",
        label="Volume",
        control=ControlType.slider,
        min_value=0.0,
        max_value=2.0,
        default_value=1.0,
    )
    return parameters


PARAMETER_CATALOG: dict[str, CatalogParameter] = _build_effect_parameters()


def catalog_widget_ids() -> list[str]:
    return sorted(WIDGET_CATALOG.keys())


def catalog_parameter_ids() -> list[str]:
    return sorted(PARAMETER_CATALOG.keys())

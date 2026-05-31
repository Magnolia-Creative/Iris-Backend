from __future__ import annotations

from app.ui_workspace.catalog import CATALOG_VERSION, PARAMETER_CATALOG, WIDGET_CATALOG
from app.ui_workspace.models import (
    LayoutNode,
    LayoutNodeType,
    ParameterControl,
    Prominence,
    ToolbarPlacement,
    UIWorkspacePlan,
    WidgetPlacement,
)


class UIWorkspaceValidationError(ValueError):
    pass


def _validate_widget_placement(placement: WidgetPlacement, *, slice_ids: set[str]) -> list[str]:
    warnings: list[str] = []
    widget = WIDGET_CATALOG.get(placement.widgetId)
    if widget is None:
        raise UIWorkspaceValidationError(f"Unknown widget id: {placement.widgetId}")
    if placement.intentSliceId not in slice_ids and placement.intentSliceId != "default":
        raise UIWorkspaceValidationError(
            f"Widget {placement.widgetId} references unknown intent slice {placement.intentSliceId}"
        )
    if placement.variant and placement.variant not in widget.variants:
        warnings.append(f"Widget {placement.widgetId} uses unknown variant {placement.variant}")
    for control in placement.controls:
        param = PARAMETER_CATALOG.get(control.parameterId)
        if param is None:
            raise UIWorkspaceValidationError(f"Unknown parameter id: {control.parameterId}")
        if control.control not in widget.valid_controls and placement.widgetId != "toolbar.parameterControls":
            warnings.append(
                f"Control {control.control} may be incompatible with widget {placement.widgetId}"
            )
        if control.minValue is not None and control.minValue < param.min_value:
            warnings.append(f"Parameter {control.parameterId} min clamped")
        if control.maxValue is not None and control.maxValue > param.max_value:
            warnings.append(f"Parameter {control.parameterId} max clamped")
    if placement.prominence == Prominence.primary and not placement.reason:
        warnings.append(f"Primary widget {placement.widgetId} should include a reason")
    return warnings


def _validate_layout_node(node: LayoutNode, *, slice_ids: set[str]) -> list[str]:
    warnings: list[str] = []
    if node.type == LayoutNodeType.widget:
        if node.widget is None:
            raise UIWorkspaceValidationError("Layout widget node missing widget placement")
        warnings.extend(_validate_widget_placement(node.widget, slice_ids=slice_ids))
        return warnings
    if node.type == LayoutNodeType.toolbar:
        return warnings
    for child in node.children:
        warnings.extend(_validate_layout_node(child, slice_ids=slice_ids))
    return warnings


def validate_workspace_plan(plan: UIWorkspacePlan) -> UIWorkspacePlan:
    slice_ids = {slice_.id for slice_ in plan.intentSlices}
    slice_ids.add("default")
    warnings: list[str] = []
    warnings.extend(_validate_layout_node(plan.layout, slice_ids=slice_ids))
    for widget in plan.toolbar.widgets:
        warnings.extend(_validate_widget_placement(widget, slice_ids=slice_ids))
    plan.warnings = list(dict.fromkeys([*plan.warnings, *warnings]))
    plan.catalogVersion = CATALOG_VERSION
    return plan


def parameter_control_from_catalog(parameter_id: str) -> ParameterControl:
    param = PARAMETER_CATALOG.get(parameter_id)
    if param is None:
        raise UIWorkspaceValidationError(f"Unknown parameter id: {parameter_id}")
    return ParameterControl(
        parameterId=param.id,
        control=param.control,
        label=param.label,
        minValue=param.min_value,
        maxValue=param.max_value,
        defaultValue=param.default_value,
    )


def toolbar_from_parameters(
    parameter_ids: list[str],
    *,
    intent_slice_id: str,
    prominence: Prominence = Prominence.primary,
) -> ToolbarPlacement:
    controls = [parameter_control_from_catalog(pid) for pid in parameter_ids]
    return ToolbarPlacement(
        widgets=[
            WidgetPlacement(
                widgetId="toolbar.parameterControls",
                variant="sliderGroup",
                prominence=prominence,
                intentSliceId=intent_slice_id,
                reason="Direct parameter tuning for active intent slice",
                controls=controls,
            )
        ],
        showNavigation=False,
        showPromptBar=False,
    )

from __future__ import annotations

from app.agent.intent.editing.models import EffectCapability, EffectCapabilityParameter


DEFAULT_EFFECT_CAPABILITIES: list[EffectCapability] = [
    EffectCapability(
        operation="addGrain",
        description="Adds film grain or analog noise texture to a clip.",
        parameters=[
            EffectCapabilityParameter(
                name="amount",
                valueType="number",
                minimum=0,
                maximum=1,
                description="Visible grain intensity.",
            )
        ],
        retrievalText="film grain noise analog texture gritty vintage old camera VHS cinematic texture organic imperfections",
        examples=[
            "make it grainy",
            "add old film texture",
            "make it feel vintage",
            "give it analog noise",
        ],
    ),
    EffectCapability(
        operation="setTemperature",
        description="Adjusts color warmth or coolness.",
        parameters=[
            EffectCapabilityParameter(
                name="value",
                valueType="number",
                minimum=-1,
                maximum=1,
                description="Negative values cool the clip, positive values warm it.",
            )
        ],
        retrievalText="warm cool temperature golden orange blue cold sunset vintage warmth film color cast",
        examples=[
            "make it warmer",
            "give it a cool blue feel",
            "make it feel vintage",
            "add golden warmth",
        ],
    ),
    EffectCapability(
        operation="setSaturation",
        description="Adjusts color intensity.",
        parameters=[
            EffectCapabilityParameter(
                name="value",
                valueType="number",
                minimum=-1,
                maximum=1,
                description="Negative values desaturate, positive values intensify color.",
            )
        ],
        retrievalText="saturation color intensity desaturated faded muted vibrant rich washed out vintage",
        examples=["make it faded", "make colors pop", "desaturate the clip", "make it look washed out"],
    ),
    EffectCapability(
        operation="setContrast",
        description="Adjusts separation between bright and dark tones.",
        parameters=[
            EffectCapabilityParameter(
                name="value",
                valueType="number",
                minimum=-1,
                maximum=1,
                description="Negative values flatten contrast, positive values increase contrast.",
            )
        ],
        retrievalText="contrast punchy flat soft dramatic faded film shadows highlights moody cinematic",
        examples=["make it more dramatic", "soften the contrast", "make it cinematic", "make it less harsh"],
    ),
    EffectCapability(
        operation="setExposure",
        description="Adjusts overall brightness.",
        parameters=[
            EffectCapabilityParameter(
                name="value",
                valueType="number",
                minimum=-1,
                maximum=1,
                description="Negative values darken, positive values brighten.",
            )
        ],
        retrievalText="exposure brightness bright dark moody dim airy overexposed underexposed light",
        examples=["make it brighter", "darken the clip", "make it moodier", "make it airy"],
    ),
    EffectCapability(
        operation="setHighlights",
        description="Adjusts bright image regions.",
        parameters=[
            EffectCapabilityParameter(
                name="value",
                valueType="number",
                minimum=-1,
                maximum=1,
                description="Negative values recover highlights, positive values lift them.",
            )
        ],
        retrievalText="highlights bright areas glow blown out soft light faded film vintage",
        examples=["soften the highlights", "make bright areas glow", "recover blown highlights"],
    ),
    EffectCapability(
        operation="setShadows",
        description="Adjusts dark image regions.",
        parameters=[
            EffectCapabilityParameter(
                name="value",
                valueType="number",
                minimum=-1,
                maximum=1,
                description="Negative values deepen shadows, positive values lift them.",
            )
        ],
        retrievalText="shadows dark areas black levels lifted faded matte moody crushed vintage film",
        examples=["lift the shadows", "make blacks look faded", "make it moodier", "crush the shadows"],
    ),
]


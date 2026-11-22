"""Tool composition: the concrete registry over the tool seam.

The seam (value types and ABCs) lives in ``app.providers.tools``; this package
holds the one concrete :class:`~app.services.tools.registry.ToolRegistry` that
composes several :class:`~app.providers.tools.base.ToolSource` s into a single
catalog the generation loop offers to the model.
"""

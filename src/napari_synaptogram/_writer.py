"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#writers

Replace code below according to your needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import tifffile
from qtpy.QtWidgets import QFileDialog

if TYPE_CHECKING:
    import napari

    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]


def save_synaptogram(viewer: napari.viewer.Viewer):
    if len(viewer.layers) == 0:
        return
    for layer in viewer.layers:
        if layer.source.path is not None:
            source = Path(layer.source.path).with_suffix(".syn")
            source = str(source)
            break
    else:
        source = None
    path, _ = QFileDialog.getSaveFileName(
        caption="Save synaptogram", dir=source, filter="Synaptograms (*.syn)"
    )
    if path:
        layer_data = [layer.as_layer_data_tuple() for layer in viewer.layers]
        write_multiple(path, layer_data)


def write_multiple(path: str, data: list[FullLayerData]) -> list[str]:
    """Writes multiple layers of different types.

    Parameters
    ----------
    path : str
        A string path indicating where to save the data file(s).
    data : A list of layer tuples.
        Tuples contain three elements: (data, meta, layer_type)
        `data` is the layer data
        `meta` is a dictionary containing all other metadata attributes
        from the napari layer (excluding the `.data` layer attribute).
        `layer_type` is a string, eg: "image", "labels", "surface", etc.

    Returns
    -------
    [path] : A list containing (potentially multiple) string paths to the saved file(s).
    """
    image = []
    image_metadata = []
    metadata = {}
    for d, md, lt in data:
        if lt == "image":
            image.append(d[..., np.newaxis])
            image_metadata.append(md)
            metadata.setdefault("name", []).append(md["name"])
            metadata.setdefault("colormap", []).append(md["colormap"]["name"])
            metadata.setdefault("scale", []).append(md["scale"])
            metadata.setdefault("visible", []).append(md["visible"])
        elif lt == "points":
            p = pd.DataFrame(d, columns=["x", "y", "z"])
            points = metadata.setdefault("points", {})
            points[md["name"]] = {
                "data": p.to_csv(index=False),
                "scale": md["scale"],
                "size": int(md["size"][0]),
                "symbol": md["symbol"][0].value,
                "out_of_slice_display": md["out_of_slice_display"],
                "visible": md["visible"],
            }
        elif lt == "shapes":
            shapes = metadata.setdefault("shapes", {})
            vertices = {}
            for i, shape in enumerate(d):
                df = pd.DataFrame(shape, columns=["x", "y", "z"])
                vertices[i] = df.rename_axis("vertex")
            if len(vertices) == 0:
                # Handle the no shapes case
                vertices = pd.DataFrame(columns=["x", "y", "z"])
                vertices.index = pd.MultiIndex(
                    levels=[[], []],
                    codes=[[], []],
                    names=["shape", "vertex"],
                )
            else:
                vertices = pd.concat(vertices, names=["shape"])
            shapes[md["name"]] = {
                "data": vertices.to_csv(index=True),
                "scale": md["scale"],
                "shape_type": md["shape_type"],
                "visible": md["visible"],
            }

    image = np.concatenate(image, axis=-1)
    tifffile.imwrite(path, image, metadata=metadata)
    return [path]

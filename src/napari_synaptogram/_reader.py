"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""

import json
from io import StringIO
from pathlib import Path

import pandas as pd
import tifffile
from synaptogram import reader

CHANNEL_CONFIG = {
    "CtBP2": {"display_color": "#FF0000"},
    "MyosinVIIa": {"display_color": "#0000FF"},
    "GluR2": {"display_color": "#00FF00"},
    "GlueR2": {"display_color": "#00FF00"},
    "PMT": {"display_color": "#FFFFFF"},
    "DAPI": {"display_color": "#FFFFFF", "visible": False},
    # Channels are tagged as unknown if there's difficulty parsing the channel
    # information from the file.
    "Unknown 1": {"display_color": "#FF0000"},
    "Unknown 2": {"display_color": "#00FF00"},
    "Unknown 3": {"display_color": "#0000FF"},
    "Unknown 4": {"display_color": "#FFFFFF"},
}


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    if path.endswith(".ims"):
        return ims_reader_function
    elif path.endswith(".syn"):
        return syn_reader_function
    return None


def ims_reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    data = []
    for path in paths:
        path = Path(path)
        fh = reader.ImarisReader(path)
        metadata = {"filename": path}
        # axes_order = [1, 0, 2, 3]
        # img = fh.image.copy()[axes_order]
        img = fh.image.copy()
        # contrast_limits = np.percentile(img, [1, 99], axis=[0, 1, 2]).T / 255
        #'contrast_limits': contrast_limits.tolist(),
        # names = [path.stem + ' ' + c['name'] for c in fh.channel_names]
        names = [c["name"] for c in fh.channel_names]
        metadata = {
            "name": names,
            "colormap": ["green", "red", "blue"],
            "scale": fh.image_info["voxel_size"],
            "channel_axis": -1,
            "axis_labels": ["X", "Y", "Z"],
        }
        data.append((img, metadata, "image"))

    return data


def syn_reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings)
    paths = [path] if isinstance(path, str) else path

    data = []
    for path in paths:
        with tifffile.TiffFile(path) as fh:
            metadata = json.loads(fh.pages[0].description)

            # Load the image
            image = fh.asarray()
            keys = ["name", "colormap", "scale", "visible"]
            image_md = {k: metadata[k] for k in keys}
            image_md.update(
                {
                    "channel_axis": -1,
                    "axis_labels": ["X", "Y", "Z"],
                }
            )
            data.append((image, image_md, "image"))

            # Load the points
            for name, points_md in metadata.get("points", {}).items():
                df = pd.read_csv(StringIO(points_md.pop("data")))
                points_md["name"] = name
                data.append((df.values, points_md, "points"))

            # Load the shapes
            for name, shapes_md in metadata.get("shapes", {}).items():
                df = pd.read_csv(
                    StringIO(shapes_md.pop("data")), index_col=[0, 1]
                )
                vertices = [v.values for _, v in df.groupby("shape")]
                shapes_md["name"] = name
                data.append((vertices, shapes_md, "shapes"))

    return data

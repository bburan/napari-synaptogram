import napari
import numpy as np
import scipy as sp
from magicgui.widgets import CheckBox, Container, PushButton, create_widget
from napari.layers import Image, Points
from skimage.draw import polygon2mask
from skimage.feature import blob_log
from skimage.util import img_as_float


class CtBP2Detection(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        # use create_widget to generate widgets from type annotations
        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )
        self._roi_layer_combo = create_widget(
            label="ROI", annotation="napari.layers.Shapes"
        )
        self._threshold_slider = create_widget(
            label="Threshold", annotation=float, widget_type="FloatSlider"
        )

        self._xy_button = PushButton(text="XY")
        self._xz_button = PushButton(text="XZ")
        self._yz_button = PushButton(text="YZ")
        self._auto_contrast_button = PushButton(text="Auto Contrast")
        self._max_proj_checkbox = CheckBox(text="Max. Proj.", value=False)
        self._xy_button.clicked.connect(lambda: self._update_dims([2, 1, 0]))
        self._xz_button.clicked.connect(lambda: self._update_dims([0, 2, 1]))
        self._yz_button.clicked.connect(lambda: self._update_dims([1, 2, 0]))
        self._max_proj_checkbox.changed.connect(self._update_projection)
        self._auto_contrast_button.clicked.connect(self._auto_contrast)

        row = [
            self._xy_button,
            self._xz_button,
            self._yz_button,
            self._auto_contrast_button,
            self._max_proj_checkbox,
        ]
        self._xyz_container = Container(widgets=row, layout="horizontal")

        self._mask_button = PushButton(text="Mask")
        self._mask_button.clicked.connect(self._mask)
        row = [
            self._image_layer_combo,
            self._roi_layer_combo,
            self._mask_button,
        ]
        self._image_container = Container(widgets=row, layout="vertical")

        self._run_button = PushButton(text="Run")
        self._run_button.clicked.connect(self._detect_points)
        self._threshold_slider.min = 0
        self._threshold_slider.max = 1
        self._threshold_slider.value = 0.1

        row = [self._threshold_slider, self._run_button]
        self._process_container = Container(widgets=row, layout="horizontal")

        # append into/extend the container with your widgets
        self.extend(
            [
                self._xyz_container,
                self._image_container,
                self._process_container,
            ]
        )
        self._viewer.layers.events.inserted.connect(
            self._rescan_layers, position="last"
        )
        self._viewer.layers.events.removed.connect(
            self._rescan_layers, position="last"
        )
        self._handling_points = False

    def _rescan_layers(self):
        self._roi_map = {}
        for layer in self._viewer.layers:
            if isinstance(layer, Points):
                layer.mouse_drag_callbacks.append(self._mouse_click)
            if isinstance(layer, Image) and "masked" not in layer.name:
                self._roi_map[layer] = None

        # Now, reconnect any masked layers that may exist.
        for src_layer in self._roi_map:
            mask_name = src_layer.name + " masked"
            for layer in self._viewer.layers:
                if layer.name == mask_name:
                    self._roi_map[src_layer] = layer
                    break

            # Make sure we set values accordingly
            # if src_layer.name.lower().endswith("ctbp2"):
            #    if self._roi_map[src_layer] is None:
            #        self._image_layer_combo.value = src_layer
            #    else:
            #        self._image_layer_combo.value = self._roi_map[src_layer]
        self._update_projection()

    def _auto_contrast(self):
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                layer.projection_mode = "max"
                layer.contrast_limits = np.percentile(layer.data, [0, 99.99])

    def _mask(self):
        roi_layer = self._roi_layer_combo.value
        if roi_layer is None:
            return

        for layer in self._viewer.layers[:]:
            if not isinstance(layer, napari.layers.Image):
                continue
            if layer not in self._roi_map:
                continue
            if layer in self._roi_map:
                layer.visible = False

            masked_layer = self._roi_map[layer]
            if masked_layer is None:
                masked_layer = self._viewer.add_image(
                    layer.data,
                    name=layer.name + " masked",
                    scale=layer.scale,
                    opacity=layer.opacity,
                    blending=layer.blending,
                    contrast_limits=layer.contrast_limits,
                    gamma=layer.gamma,
                    colormap=layer.colormap,
                    projection_mode=layer.projection_mode,
                    depiction=layer.depiction,
                    rendering=layer.rendering,
                )
                self._roi_map[layer] = masked_layer
                if layer.name.lower().endswith("ctbp2"):
                    self._image_layer_combo.value = masked_layer

            # Make a copy of the master layer to operate on. Then, iterate
            # through all shapes, apply the mask, then set the modified data on
            # the masked layer.
            layer_data = layer.data.copy()

            for polygon in roi_layer.data:
                # Polygons are drawn in 2D space. Determine axes of the space.
                axes = np.flatnonzero(polygon.std(axis=0))
                shape = np.take(masked_layer.data.shape, axes)
                vertices = polygon[:, axes]
                broadcast = tuple(
                    slice(None) if i in axes else np.newaxis for i in range(3)
                )
                mask = polygon2mask(shape, vertices)[broadcast]
                layer_data = layer_data * mask

            masked_layer.data = layer_data
            masked_layer.visible = True

    def _update_projection(self):
        if self._max_proj_checkbox.value:
            self._viewer.dims.thickness = [
                (r.stop - r.start) * 2 for r in self._viewer.dims.range
            ]
        else:
            self._viewer.dims.thickness = (0, 0, 0)

    def _update_dims(self, order):
        self._viewer.dims.order = order

    def _detect_points(self):
        """
        Scans for points and adds them as a new layer
        """
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return

        image = img_as_float(image_layer.data)
        threshold = self._threshold_slider.value
        points = blob_log(image, threshold=threshold, num_sigma=1)[:, :3]
        name = image_layer.name + " points"
        if name in self._viewer.layers:
            self._viewer.layers[name].data = points
        else:
            layer = self._viewer.add_points(
                points,
                name=name,
                scale=image_layer.scale,
                size=2,
                symbol="o",
                out_of_slice_display=True,
            )

        for layer in self._viewer.layers:
            if not isinstance(layer, napari.layers.Image):
                continue
            if layer != image_layer:
                layer.visible = False

    def _mouse_click(self, layer, event):
        if layer.mode != "pan_zoom":
            # A tool (e.g., add, remove, select) is being used. Don't interfere
            # with what's already going on otherwise we may end up with
            # duplicate points or delete two points.
            return
        if event.buttons[0] == 2:
            if "Shift" in event.modifiers:
                self._remove_point(layer, event)
            else:
                self._add_point(layer, event)

    def _remove_point(self, layer, event):
        # This seems to handle hit testing. Not thrilled about using an
        # internal API, but it's styled after
        # napari.layers.points._points_mouse_bindings.
        point_index = layer._get_value_(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        if point_index is not None:
            layer.pop(point_index)

    def _add_point(self, layer, event):
        image_layer = self._image_layer_combo.value
        if self._viewer.dims.ndisplay == 2:
            # Logic for handling 2D view.
            near_point = list(event.position)
            far_point = list(event.position)

            # Find the axis to project the ray along.
            ray_axis = ({0, 1, 2} - set(self._viewer.dims.displayed)).pop()

            # Get the thickness of the view. The thickness is the full range
            # (lower to upper), but point click is in the center.
            thickness = self._viewer.dims.thickness[ray_axis] / 2
            near_point[ray_axis] += thickness
            far_point[ray_axis] -= thickness
            near_point = image_layer.world_to_data(near_point)
            far_point = image_layer.world_to_data(far_point)
            if thickness == 0:
                layer.add(near_point)
                return
        else:
            # Logic for handling 3D view.
            # Find coordinates where ray enters/exists layer bounding box.
            near_point, far_point = image_layer.get_ray_intersections(
                event.position, event.view_direction, event.dims_displayed
            )
            if (near_point is None) or (far_point is None):
                return

        # Calculate intensities along a ray that passes through the layer
        # bounding box. Find the coordinate of the maximum intensity along this
        # and define this as the location of the new point to add to the points
        # layer.
        ray = np.linspace(near_point, far_point, 100, endpoint=True)
        intensities = sp.ndimage.map_coordinates(
            image_layer.data,
            ray.T,
            mode="constant",
            cval=0,
        )
        layer.add(ray[intensities.argmax()])

import napari
import numpy as np
import scipy as sp
from magicgui.widgets import CheckBox, Container, PushButton, create_widget
from napari.layers import Image, Points, Shapes
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

        self._crop_button = PushButton(text="Crop")
        self._crop_button.clicked.connect(self._crop)
        self._crop_container = Container(
            widgets=[self._crop_button],
            layout="horizontal",
        )

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
                self._crop_container,
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

    def _find_last_rectangle(self):
        """Return (Shapes layer, shape index) for the most recently drawn
        rectangle across all Shapes layers, or (None, None) if there is none.
        """
        for layer in reversed(list(self._viewer.layers)):
            if not isinstance(layer, Shapes):
                continue
            types = list(layer.shape_type)
            for idx in range(len(types) - 1, -1, -1):
                if types[idx] == "rectangle":
                    return layer, idx
        return None, None

    def _get_rectangle_geometry(self):
        """Geometry of the most-recently drawn rectangle, in world coords.

        Reads the rectangle's vertices from layer.data and applies the
        Shapes layer's full data->world transform chain. This is important
        because napari's "transform" mode on the layer rotates the displayed
        shape via layer.rotate / layer.affine without touching layer.data
        — reading .data alone would miss that rotation.

        Returns None if no rectangle exists. Otherwise returns a dict:

            vertices : (4, 2) world coords in (axes[0], axes[1]) order,
                in perimeter order — v[1]-v[0] and v[3]-v[0] are the two
                perpendicular edge vectors.
            vertices_data : (4, ndim) raw layer.data vertices (pre-transform),
                in case the caller needs them.
            axes : list[int] the two world axes the rectangle spans
                (the third has 0 variance across vertices).
            center : (2,) centroid in (axes[0], axes[1]) world coords.
            width, height : float lengths of the v[1]-v[0] and v[3]-v[0] edges.
            long_length, short_length : max/min of (width, height).
            angle : rotation of the LONG edge from +axes[0], degrees,
                normalized to (-90, 90].
            bbox_min, bbox_max : (2,) axis-aligned bbox of the rotated
                rectangle in (axes[0], axes[1]) world coords (raw float).
        """
        shapes_layer, idx = self._find_last_rectangle()
        if shapes_layer is None:
            return None

        vertices_data = np.asarray(shapes_layer.data[idx], dtype=float)
        # Apply layer.scale/rotate/translate/affine so layer-level rotation
        # (napari's transform mode) is reflected.
        rect = np.asarray(
            shapes_layer._transforms[1:3].simplified(vertices_data),
            dtype=float,
        )

        axes = sorted(np.argsort(rect.std(axis=0))[-2:].tolist())
        v = rect[:, axes]

        e1 = v[1] - v[0]
        e3 = v[3] - v[0]
        width = float(np.linalg.norm(e1))
        height = float(np.linalg.norm(e3))
        if width >= height:
            long_edge, long_length, short_length = e1, width, height
        else:
            long_edge, long_length, short_length = e3, height, width

        angle = float(np.degrees(np.arctan2(long_edge[1], long_edge[0])))
        if angle > 90:
            angle -= 180
        elif angle <= -90:
            angle += 180

        return {
            "vertices": v,
            "vertices_data": vertices_data,
            "axes": axes,
            "center": v.mean(axis=0),
            "width": width,
            "height": height,
            "long_length": long_length,
            "short_length": short_length,
            "angle": angle,
            "bbox_min": v.min(axis=0),
            "bbox_max": v.max(axis=0),
        }

    def _crop(self):
        geom = self._get_rectangle_geometry()
        if geom is None:
            return
        shapes_layer, idx = self._find_last_rectangle()

        # Vertices in world coords (full ndim — includes the slicing axis,
        # which is constant across all four), so we can map them into each
        # Image layer's own data space below.
        world_full = np.asarray(
            shapes_layer._transforms[1:3].simplified(
                np.asarray(shapes_layer.data[idx], dtype=float)
            )
        )

        for layer in self._viewer.layers:
            if not isinstance(layer, Image):
                continue
            data = layer.data
            ndim = data.ndim

            # Rectangle vertices in this image's own data coords.
            v_img = np.array([layer.world_to_data(p) for p in world_full])

            # The two axes the rectangle spans (the third has 0 variance).
            img_axes = sorted(np.argsort(v_img.std(axis=0))[-2:].tolist())
            vp = v_img[:, img_axes]

            # napari stores rectangles as 4 vertices in perimeter order, so
            # vp[1]-vp[0] and vp[3]-vp[0] are the two perpendicular edges.
            e1 = vp[1] - vp[0]
            e3 = vp[3] - vp[0]
            L1 = float(np.linalg.norm(e1))
            L3 = float(np.linalg.norm(e3))
            if L1 < 1 or L3 < 1:
                continue
            if L3 >= L1:
                e_long, e_short, L_long, L_short = e3, e1, L3, L1
            else:
                e_long, e_short, L_long, L_short = e1, e3, L1, L3
            long_hat = e_long / L_long
            short_hat = e_short / L_short

            out_long = int(round(L_long))
            out_short = int(round(L_short))

            # Affine: input_coord = M @ output_coord + offset.
            # Non-plane axes pass through unchanged. Output index along
            # img_axes[0] (canvas X) traverses the long edge of the rectangle
            # in the input; along img_axes[1] (canvas Y) traverses the short
            # edge. The long edge therefore ends up along the output X axis.
            M = np.zeros((ndim, ndim))
            offset = np.zeros(ndim)
            for ax in range(ndim):
                if ax not in img_axes:
                    M[ax, ax] = 1.0
            M[img_axes[0], img_axes[0]] = long_hat[0]
            M[img_axes[0], img_axes[1]] = short_hat[0]
            M[img_axes[1], img_axes[0]] = long_hat[1]
            M[img_axes[1], img_axes[1]] = short_hat[1]
            offset[img_axes[0]] = vp[0, 0]
            offset[img_axes[1]] = vp[0, 1]

            output_shape = list(data.shape)
            output_shape[img_axes[0]] = out_long
            output_shape[img_axes[1]] = out_short

            layer.data = sp.ndimage.affine_transform(
                data,
                M,
                offset=offset,
                output_shape=output_shape,
                order=1,
            )

        # The rectangle's coords no longer correspond to the new image data.
        shapes_layer.selected_data = {idx}
        shapes_layer.remove_selected()

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

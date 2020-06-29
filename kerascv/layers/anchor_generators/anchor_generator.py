# Copyright 2020 The Keras CV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


class AnchorGenerator(tf.keras.layers.Layer):
    """Defines a AnchorGenerator that generates anchor boxes for a single feature map.

    # Attributes:
        scales: A list/tuple of positive floats (usually less than 1.) as a fraction to shorter side of `image_size`.
            It represents the base anchor size (when aspect ratio is 1.). For example, if `image_size` is (300, 200),
            and `scales=[.1]`, then the base anchor size is 20.
        aspect_ratios: a list/tuple of positive floats representing the ratio of anchor width to anchor height.
            **Must** have the same length as `scales`. For example, if `image_size=(300, 200)`, `scales=[.1]`,
            and `aspect_ratios=[.64]`, the base anchor size is 20, then anchor height is 25 and anchor width is 16.
            The anchor aspect ratio is independent to the original aspect ratio of image size.
        anchor_dimensions: A single int, or a list/tuple of ints. It represents the anchor dimension. If not None,
            the `scales` are fraction of anchor_dimensions instead of fraction of `image_size`.
        stride: A list/tuple of 2 ints or floats representing the distance between anchor points.
            For example, `stride=(30, 40)` means each anchor is separated by 30 pixels in height, and
            40 pixels in width. Defaults to `None`, where anchor stride would be calculated as
            `min(image_height, image_width) / feature_map_height` and
            `min(image_height, image_width) / feature_map_width`.
        offset: A list/tuple of 2 floats between [0., 1.] representing the center of anchor points relative to
            the upper-left border of each feature map cell. Defaults to `None`, which is the center of each
            feature map cell when `stride=None`, or center of anchor stride otherwise.
        clip_boxes: Boolean to represents whether the anchor coordinates should be clipped to the image size.
            Defaults to `True`.
        normalize_coordinates: Boolean to represents whether the anchor coordinates should be normalized to [0., 1.]
            with respect to the image size. Defaults to `True`.

    """

    def __init__(
        self,
        scales,
        aspect_ratios,
        anchor_dimensions=None,
        stride=None,
        offset=None,
        clip_boxes=True,
        normalize_coordinates=True,
        name=None,
        **kwargs
    ):
        """Constructs a AnchorGenerator."""

        self.scales = scales
        self.anchor_dimensions = anchor_dimensions
        # whether use ratio with image size, or areas.
        self.scales_on_image = (anchor_dimensions is None)
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        self.offset = offset
        self.clip_boxes = clip_boxes
        self.normalize_coordinates = normalize_coordinates
        super(AnchorGenerator, self).__init__(name=name, **kwargs)

    def call(self, image_size, feature_map_size):
        feature_map_height = tf.cast(feature_map_size[0], dtype=tf.float32)
        feature_map_width = tf.cast(feature_map_size[1], dtype=tf.float32)
        image_height = tf.cast(image_size[0], dtype=tf.float32)
        image_width = tf.cast(image_size[1], dtype=tf.float32)

        min_image_size = tf.minimum(image_width, image_height)

        if self.stride is None:
            stride_height = tf.cast(
                min_image_size / feature_map_height, dtype=tf.float32
            )
            stride_width = tf.cast(min_image_size / feature_map_width, dtype=tf.float32)
        else:
            stride_height = tf.cast(self.stride[0], dtype=tf.float32)
            stride_width = tf.cast(self.stride[1], dtype=tf.float32)

        if self.offset is None:
            offset_height = tf.constant(0.5, dtype=tf.float32)
            offset_width = tf.constant(0.5, dtype=tf.float32)
        else:
            offset_height = tf.cast(self.offset[0], dtype=tf.float32)
            offset_width = tf.cast(self.offset[1], dtype=tf.float32)

        len_k = len(self.aspect_ratios)
        aspect_ratios_sqrt = tf.cast(tf.sqrt(self.aspect_ratios), tf.float32)
        scales = tf.cast(self.scales, dtype=tf.float32)
        if self.scales_on_image:
            anchor_dimension = min_image_size
        else:
            anchor_dimension = tf.cast(self.anchor_dimensions, tf.float32)
        # [1, 1, K]
        anchor_heights = tf.reshape(
            (scales / aspect_ratios_sqrt) * anchor_dimension, (1, 1, -1)
        )
        anchor_widths = tf.reshape(
            (scales * aspect_ratios_sqrt) * anchor_dimension, (1, 1, -1)
        )

        # [W]
        cx = (tf.range(feature_map_width) + offset_width) * stride_width
        # [H]
        cy = (tf.range(feature_map_height) + offset_height) * stride_height
        # [H, W]
        cx_grid, cy_grid = tf.meshgrid(cx, cy)
        # [H, W, 1]
        cx_grid = tf.expand_dims(cx_grid, axis=-1)
        cy_grid = tf.expand_dims(cy_grid, axis=-1)
        # [H, W, K]
        cx_grid = tf.tile(cx_grid, (1, 1, len_k))
        cy_grid = tf.tile(cy_grid, (1, 1, len_k))
        # [H, W, K]
        anchor_heights = tf.tile(
            anchor_heights, (feature_map_height, feature_map_width, 1)
        )
        anchor_widths = tf.tile(
            anchor_widths, (feature_map_height, feature_map_width, 1)
        )

        # [H, W, K, 2]
        box_centers = tf.stack([cy_grid, cx_grid], axis=3)
        # [H * W * K, 2]
        box_centers = tf.reshape(box_centers, [-1, 2])
        # [H, W, K, 2]
        box_sizes = tf.stack([anchor_heights, anchor_widths], axis=3)
        # [H * W * K, 2]
        box_sizes = tf.reshape(box_sizes, [-1, 2])
        # y_min, x_min, y_max, x_max
        # [H * W * K, 4]
        box_tensor = tf.concat(
            [box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1
        )

        if self.clip_boxes:
            y_min, x_min, y_max, x_max = tf.split(
                box_tensor, num_or_size_splits=4, axis=1
            )
            y_min_clipped = tf.maximum(tf.minimum(y_min, image_height), 0)
            y_max_clipped = tf.maximum(tf.minimum(y_max, image_height), 0)
            x_min_clipped = tf.maximum(tf.minimum(x_min, image_width), 0)
            x_max_clipped = tf.maximum(tf.minimum(x_max, image_width), 0)
            box_tensor = tf.concat(
                [y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped], axis=1
            )

        if self.normalize_coordinates:
            y_min, x_min, y_max, x_max = tf.split(
                box_tensor, num_or_size_splits=4, axis=1
            )
            y_min_norm = y_min / image_height
            y_max_norm = y_max / image_height
            x_min_norm = x_min / image_width
            x_max_norm = x_max / image_width
            box_tensor = tf.concat(
                [y_min_norm, x_min_norm, y_max_norm, x_max_norm], axis=1
            )

        return box_tensor

    def get_config(self):
        config = {
            "scales": self.scales,
            "aspect_ratios": self.aspect_ratios,
            "anchor_dimensions": self.anchor_dimensions,
            "stride": self.stride,
            "offset": self.offset,
            "clip_boxes": self.clip_boxes,
            "normalize_coordinates": self.normalize_coordinates,
        }
        base_config = super(AnchorGenerator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

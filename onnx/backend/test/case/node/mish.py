# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Mish(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Mish',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        x = x * np.tanh(np.log(np.exp(x) + 1))  # expected output array([-0.30340144,  0. ,  0.86509836], dtype=float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_softplus_example')

        x = np.array([1, 1]).astype(np.float32)
        y = np.log(np.exp(x) + 1)                 # expected array([0.86509836, 0.86509836], dtype=float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_softplus')

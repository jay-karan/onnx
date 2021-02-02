# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Opmul(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Opmul',
            inputs=['X'],
            outputs=['Y'],
        )

        input_data = np.array([1.0,2.0,3.0], dtype=np.float32)

        # Calculate expected output data
    
        expected_output = np.array([1.0,4.0,9.0], dtype=np.float32)

        expect(node, inputs=[input_data], outputs=[expected_output],
               name='test_opmul')

# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from emukit.quadrature.acquisitions import SquaredCorrelation
from emukit.quadrature.methods import VanillaBayesianQuadrature
import numpy as np
from typing import Tuple


class RayAcquisition(SquaredCorrelation):
    def __init__(self, model: VanillaBayesianQuadrature, v_init, num_timesteps):
        super().__init__(model)
        self.v_init = v_init
        self.num_timesteps = num_timesteps

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the acquisition function at x.

        :param x: The locations where to evaluate, shape (n_points, input_dim) .
        :return: The acquisition values at x, shape (n_points, 1).
        """
        #return self._evaluate(x)[0]
        # loop over timesteps and evaluate at each step
        vs_ray_new = np.linspace(self.v_init, x, self.num_timesteps)[1:]
        
        eval_sum = 0
        for i in range(self.num_timesteps-1):
            eval_sum += self._evaluate(vs_ray_new[i])[0]
        return eval_sum/(self.num_timesteps-1)
        #return self._evaluate(vs_ray_new)[0]

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the acquisition function and its gradient.

        :param x: The locations where to evaluate, shape (n_points, input_dim).
        :return: The acquisition values and corresponding gradients at x,
                 shapes (n_points, 1) and (n_points, input_dim)
        """
        # value
        vs_ray_new = np.linspace(self.v_init, x, self.num_timesteps)[1:]
        for i in range(self.num_timesteps-1):
            squared_correlation, squared_correlation_gradient = super().evaluate_with_gradients(vs_ray_new[i])
            if i == 0:
                eval_sum = squared_correlation
                grad_sum = squared_correlation_gradient
            else:
                eval_sum += squared_correlation
                grad_sum += squared_correlation_gradient
        return eval_sum/(self.num_timesteps-1), grad_sum/(self.num_timesteps-1)
        #return squared_correlation, squared_correlation_gradient


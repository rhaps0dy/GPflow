# Copyright 2017 Artem Artemev @awav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..core.errors import GPflowError
from ..core.compilable import Build
from ..models.model import Model

from . import optimizer
from . import external_optimizer


class ScipyOptimizer(optimizer.Optimizer):
    def __init__(self, **kwargs):
        self._optimizer_kwargs = kwargs
        self._optimizer = None
        self._model = None
    

    def make_optimize_tensor(self, model, session=None, var_list=None, **kwargs):
        """
        Make SciPy optimization tensor.
        The `make_optimize_tensor` method builds optimization tensor and initializes
        all necessary variables created by optimizer.

            :param model: GPflow model.
            :param session: Tensorflow session.
            :param var_list: List of variables for training.
            :param kwargs: Scipy optional optimization parameters,
                - `maxiter`, maximal number of iterations to perform.
                - `disp`, if True, prints convergence messages.
            :return: Tensorflow operation.
        """
        session = model.enquire_session(session)
        with session.as_default():
            var_list = self._gen_var_list(model, var_list)
            optimizer_kwargs = self._optimizer_kwargs.copy()
            if 'options' in optimizer_kwargs:
                optimizer_kwargs['options'].update(kwargs)
            objective = model.objective
            optimizer = external_optimizer.ScipyOptimizerInterface(
                objective, var_list=var_list, **optimizer_kwargs)
            model.initialize(session=session)
            return optimizer

    def minimize(self, model, session=None, var_list=None, feed_dict=None,
                 maxiter=1000, initialize=False, anchor=True, **kwargs):
        """
        Minimizes objective function of the model.

        :param model: GPflow model with objective tensor.
        :param session: Session where optimization will be run.
        :param var_list: List of extra variables which should be trained during optimization.
        :param feed_dict: Feed dictionary of tensors passed to session run method.
        :param maxiter: Number of run interation. Note: scipy optimizer can do early stopping
            if model converged.
        :param initialize: If `True` model parameters will be re-initialized even if they were
            initialized before for gotten session.
        :param anchor: If `True` trained parameters computed during optimization at
            particular session will be synchronized with internal parameter values.
        :param kwargs: This is a dictionary of extra parameters for session run method and
            one `disp` option which will be passed to scipy optimizer.
        """
        if model is None or not isinstance(model, Model):
            raise ValueError('Unknown type passed for optimization.')

        if model.is_built_coherence() is Build.NO:
            raise GPflowError('Model is not built.')

        session = model.enquire_session(session)
        self._model = model
        disp = kwargs.pop('disp', False)
        optimizer = self.make_optimize_tensor(model, session,
            var_list=var_list, maxiter=maxiter, disp=disp)
        self._optimizer = optimizer
        feed_dict = self._gen_feed_dict(model, feed_dict)
        optimizer.minimize(session=session, feed_dict=feed_dict, **kwargs)
        if anchor:
            model.anchor(session)

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

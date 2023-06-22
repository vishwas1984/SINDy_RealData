import math
import torch
import gpytorch
from gpytorch.constraints import Positive
from gpytorch.constraints import Interval

class MaternAdditiveKernel(gpytorch.kernels.Kernel):
    is_stationary = True # the custom kernel is stationary
    has_lengthscale = True

    # register the parameter when initializing the kernel
    def __init__(self, weights_prior=None, weights_constraint=None,
                 lengthscale_unique = False,
                 lengthscales_prior=None, lengthscales_constraint=None,
                 **kwargs):
        self.lengthscale_unique = lengthscale_unique
        super().__init__(**kwargs)
        
        n_baseker = 3
        
        # register the raw parameter
        self.register_parameter(
                name='raw_weights', parameter=torch.nn.Parameter( torch.zeros(*self.batch_shape, n_baseker))
        )
        self.register_parameter(
                name='raw_lengthscales', parameter=torch.nn.Parameter( torch.zeros(*self.batch_shape, n_baseker))
        )
        
        if weights_constraint is None:
            weights_constraint = Interval( torch.zeros(n_baseker), torch.ones(n_baseker) )
            #weights_constraint = Positive()
        if lengthscales_constraint is None:
            lengthscales_constraint = Positive()

        # register the constraint
        self.register_constraint('raw_weights', weights_constraint)
        self.register_constraint('raw_lengthscales', lengthscales_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if weights_prior is not None:
            self.register_prior(
                "weights_prior",
                weights_prior,
                lambda m: m.weights,
                lambda m, v : m._set_weights(v),
            )
        if lengthscales_prior is not None:
            self.register_prior(
                "lengthscales_prior",
                lengthscales_prior,
                lambda m: m.lengthscales,
                lambda m, v : m._set_lengthscales(v),
            )
    
    # now set up the 'actual' paramter
    @property
    def weights(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_weights_constraint.transform(self.raw_weights)

    @weights.setter
    def weights(self, value):
        return self._set_weights(value)

    def _set_weights(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weights)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_weights=self.raw_weights_constraint.inverse_transform(value))
    
    @property
    def lengthscales(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_lengthscales_constraint.transform(self.raw_lengthscales)

    @lengthscales.setter
    def lengthscales(self, value):
        return self._set_lengthscales(value)

    def _set_lengthscales(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscales)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_lengthscales=self.raw_lengthscales_constraint.inverse_transform(value))
    
    
    # this is the kernel function
    def forward(self, x1, x2, diag=False, **params):
        
        weights_sum = torch.sum(self.weights)
        w0 = self.weights[0] / weights_sum
        w1 = self.weights[1] / weights_sum
        w2 = self.weights[2] / weights_sum

        if self.lengthscale_unique:
            K = w0 * gpytorch.kernels.MaternKernel(nu=0.5).forward(x1=x1, x2=x2) \
            + w1 * gpytorch.kernels.MaternKernel(nu=1.5).forward(x1=x1, x2=x2) \
            + w2 * gpytorch.kernels.MaternKernel(nu=2.5).forward(x1=x1, x2=x2)
            
        else:

            ###############################################
            #### Matern kernel ##########################
            ###############################################
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

            x1_matern12 = (x1 - mean).div(self.lengthscales[0])
            x2_matern12 = (x2 - mean).div(self.lengthscales[0])
            distance12 = self.covar_dist(x1_matern12, x2_matern12, diag=diag, **params)
            exp_component12 = torch.exp(-math.sqrt(0.5 * 2) * distance12)
            constant_component12 = 1

            x1_matern32 = (x1 - mean).div(self.lengthscales[1])
            x2_matern32 = (x2 - mean).div(self.lengthscales[1])
            distance32 = self.covar_dist(x1_matern32, x2_matern32, diag=diag, **params)
            exp_component32 = torch.exp(-math.sqrt(1.5 * 2) * distance32)
            constant_component32 = (math.sqrt(3) * distance32).add(1)

            x1_matern52 = (x1 - mean).div(self.lengthscales[2])
            x2_matern52 = (x2 - mean).div(self.lengthscales[2])
            distance52 = self.covar_dist(x1_matern52, x2_matern52, diag=diag, **params)
            exp_component52 = torch.exp(-math.sqrt(2.5 * 2) * distance52)
            constant_component52 = (math.sqrt(5) * distance52).add(1).add(5.0 / 3.0 * distance52**2)

            K0 = constant_component12 * exp_component12
            K1 = constant_component32 * exp_component32
            K2 = constant_component52 * exp_component52

            K = w0*K0 + w1*K1 + w2*K2
        
        return K
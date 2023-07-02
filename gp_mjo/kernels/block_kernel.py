import torch
import gpytorch
from gpytorch.constraints import Positive
from gpytorch.constraints import Interval

from gpytorch.functions import MaternCovariance

class CustomBlockKernel(gpytorch.kernels.Kernel):
    is_stationary = True # the custom kernel is stationary
    has_lengthscale = True
    
    # register the parameter when initializing the kernel
    def __init__(self, nu=0.5, num_paras = 2, para_lower = 1e-10, para_upper=2.0,
                 #block_kernel=gpytorch.kernels.MaternKernel(nu=0.5),
                 paras_prior=None, paras_constraint=None,
                 **kwargs):
        #self.block_kernel = block_kernel
        self.nu = nu
        super().__init__(**kwargs)
        
        # register the raw parameter
        self.register_parameter(
                name='raw_paras', parameter=torch.nn.Parameter( torch.zeros(*self.batch_shape, num_paras))
        )
        
        if paras_constraint is None:
            paras_constraint = Interval( para_lower*torch.ones(num_paras), para_upper*torch.ones(num_paras) )
        
        # register the constraint
        self.register_constraint('raw_paras', paras_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if paras_prior is not None:
            self.register_prior(
                "paras_prior",
                paras_prior,
                lambda m: m.paras,
                lambda m, v : m._set_paras(v),
            )
    
    # now set up the 'actual' paramter
    @property
    def paras(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_paras_constraint.transform(self.raw_paras)

    @paras.setter
    def paras(self, value):
        return self._set_paras(value)

    def _set_paras(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_paras)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_paras=self.raw_paras_constraint.inverse_transform(value))
    
    # this is the kernel function
    def forward(self, x1, x2, alpha=None, diag=False, **params):
        
        if x1.dim() == 1 or x1.shape[0]==1:
            x1 = x1.reshape(-1,1)
        if x2.dim() == 1 or x2.shape[0]==1:
            x2 = x2.reshape(-1,1)
        
        n1 = x1.shape[0] // 2
        n2 = x2.shape[0] // 2

        x1_front = x1[:n1, :]
        x1_back = x1[n1:, :]

        x2_front = x2[:n2, :]
        x2_back = x2[n2:, :]

        if alpha is None:
            alpha = self.paras[0]

        #K22 = self.block_kernel.forward(x1=x1_back,x2=x2_back, diag=diag, **params) # n1 * n2 matrix
        K22 = MaternCovariance.apply(
            x1_back, x2_back, self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params)
        ) # n1 * n2 matrix
        B1 = alpha * torch.eye(n1) # n1 * n1 matrix
        B2 = alpha * torch.eye(n2) # n2 * n2 matrix
        K12 = B1 @ K22 # n1 * n2 matrix
        K21 = K22 @ B2.T # n1 * n2 matrix

        K12_cond = MaternCovariance.apply(
            x1_front, x2_front, self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params)
        )# n1 * n2 matrix
        K11 = K12_cond + K12 @ B2.T # n1 * n2 matrix

        K = torch.cat( (torch.cat((K11,K12),dim=1), torch.cat((K21,K22),dim=1)), dim=0) # 2(n1) * 2(n2) matrix
        K_diag = torch.diagonal(K, 0)
        if diag:
            return K_diag
        else:
            return K
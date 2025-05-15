# Copyright 2019 Ondrej Skopek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Tuple, Any

import torch
from torch import Tensor
import torch.nn.functional as F

from .common import sqrt, e_i, expand_proj_dims
from .manifold import RadiusManifold
import math

class Sphere(RadiusManifold):

    def exp_map_mu0(self, x: Tensor) -> Tensor:
        return exp_map_mu0(expand_proj_dims(x), radius=self.radius)

    def inverse_exp_map_mu0(self, x: Tensor) -> Tensor:
        return inverse_exp_map_mu0(x, radius=self.radius)

    def parallel_transport_mu0(self, x: Tensor, dst: Tensor) -> Tensor:
        return parallel_transport_mu0(x, dst, radius=self.radius)

    def inverse_parallel_transport_mu0(self, x: Tensor, src: Tensor) -> Tensor:
        return inverse_parallel_transport_mu0(x, src, radius=self.radius)

    def mu_0(self, shape: torch.Size, **kwargs: Any) -> Tensor:
        return mu_0(shape, radius=self.radius, **kwargs)

    def sample_projection_mu0(self, x: Tensor, at_point: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        return sample_projection_mu0(x, at_point, radius=self.radius)

    def inverse_sample_projection_mu0(self, x_proj: Tensor, at_point: Tensor) -> Tuple[Tensor, Tensor]:
        return inverse_sample_projection_mu0(x_proj, at_point, radius=self.radius)

    def logdet(self, mu: Tensor, std: Tensor, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        u = data[0]
        return _logdet(u, self.radius)

    @property
    def curvature(self) -> Tensor:
        return 1. / (self.radius**2)


def _logdet(u: Tensor, radius: Tensor) -> Tensor:
    assert torch.isfinite(u).all()
    # det [(\partial / \partial v) proj_{\mu}(v)] = (R|sin(r)| / r)^(n-1)
    r = torch.norm(u, dim=-1, p=2) / radius
    n = u.shape[-1] - 1

    logdet_partial = (n - 1) * (torch.log(radius) + torch.log(torch.abs(torch.sin(r)).clamp(min=1e-5)) -
                                torch.log(r.clamp(min=1e-5)))
    assert torch.isfinite(logdet_partial).all()
    return logdet_partial


def mu_0(shape: torch.Size, radius: Tensor, **kwargs: Any) -> Tensor:
    return e_i(i=0, shape=shape, **kwargs) * radius


def parallel_transport_mu0(v: Tensor, dst: Tensor, radius: Tensor) -> Tensor:
    coef = torch.sum(dst * v, dim=-1, keepdim=True) / (radius * (radius + dst[..., 0:1]))
    right = torch.cat((dst[..., 0:1] + radius, dst[..., 1:]), dim=-1)
    return v - coef * right


def inverse_parallel_transport_mu0(x: Tensor, src: Tensor, radius: Tensor) -> Tensor:
    coef = x[..., 0:1] / (radius + src[..., 0:1])
    right = torch.cat((src[..., 0:1] + radius, src[..., 1:]), dim=-1)
    return x - coef * right


#def exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
#    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True) / radius
#    x_normed = x / x_norm
#    ret = torch.cos(x_norm) * at_point + torch.sin(x_norm) * x_normed
#    assert torch.isfinite(ret).all()
#    return ret
def exp_map(x: Tensor, at_point: Tensor, radius: Tensor,
            eps: float = 1e-8) -> Tensor:
    """
    Exponential map on the sphere S^{n−1}(r) with centre `at_point`.
    Handles zero-norm and very-large tangent vectors safely.
    """
    # ‖x‖
    norm = torch.norm(x, dim=-1, keepdim=True)

    # 1. 处理零向量: exp_{μ}(0)=μ
    zero_mask = norm < eps

    # 2. 裁剪过大的‖x‖到 < π r（注: sphere 的可注射半径）
    max_norm = (math.pi - 1e-4) * radius
    clipped  = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    x_safe   = x * clipped
    norm     = torch.norm(x_safe, dim=-1, keepdim=True)  # 重新计算

    # 3. 方向 & 执行映射
    direction = x_safe / (norm + eps)
    theta     = norm / radius
    ret       = torch.cos(theta) * at_point + torch.sin(theta) * direction * radius

    # 4. 把零-向量位置替换回 at_point
    ret = torch.where(zero_mask, at_point, ret)

    # 再做一次保险
    ret = torch.nan_to_num(ret, nan=0.0, posinf=1.0, neginf=-1.0)
    return ret

#def exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
#    assert x[..., 0].allclose(torch.zeros_like(x[..., 0]))
#    x = x[..., 1:]
#    x_norm = torch.norm(x, p=2, keepdim=True, dim=-1) / radius
#    x_normed = F.normalize(x, p=2, dim=-1) * radius
#    ret = torch.cat((torch.cos(x_norm) * radius, torch.sin(x_norm) * x_normed), dim=-1)
#    assert torch.isfinite(ret).all()
#    return ret

def exp_map_mu0(x: Tensor,
                radius: Tensor,
                eps: float = 1e-8) -> Tensor:
    """
    Exponential map at the north pole μ₀ = (r,0,…,0) on S^{n−1}(r).

    Args
    ----
    x      : (..., n)  tangent vector, 其第一维必须为 0
    radius : 标量或 broadcast-compatible 张量
    """
    # 保证第一坐标为 0（在北极的切空间）
    assert torch.allclose(x[..., 0], torch.zeros_like(x[..., 0]), atol=1e-6)

    # 去掉第一维，剩余 (n-1) 个坐标
    tangent = x[..., 1:]

    # ‖x‖
    norm = torch.norm(tangent, p=2, dim=-1, keepdim=True)

    # ===== 数值稳健处理 =====
    # 1) 裁剪过大的范数以避免注射半径之外
    max_norm = (math.pi - 1e-4) * radius
    clipped  = torch.where(norm > max_norm, max_norm / (norm + eps), torch.ones_like(norm))
    tangent  = tangent * clipped
    norm     = torch.norm(tangent, p=2, dim=-1, keepdim=True)   # 重新计算

    # 2) 单位方向；加 eps 防止 0 除
    direction = tangent / (norm + eps)

    # 3) 指数映射
    theta  = norm / radius                    # (...,1)
    first  = torch.cos(theta) * radius        # (...,1)
    others = torch.sin(theta) * direction * radius
    ret    = torch.cat((first, others), dim=-1)

    # 4) 零向量：exp_{μ₀}(0)=μ₀
    zero_mask = norm < eps
    if zero_mask.any():
        north_pole = torch.zeros_like(ret)
        north_pole[..., 0] = radius.squeeze()  # broadcast 兼容
        ret = torch.where(zero_mask, north_pole, ret)

    # 5) 最后一层保险，去除 nan/inf
    ret = torch.nan_to_num(ret, nan=0.0, posinf=1.0, neginf=-1.0)
    return ret



def inverse_exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    alpha = torch.sum(at_point * x, dim=-1, keepdim=True) / (radius**2)
    coef = torch.acos(torch.clamp(alpha, min=-1., max=1.)) / sqrt(1. - alpha**2)
    ret = coef * (x - alpha * at_point)
    assert torch.isfinite(ret).all()
    return ret


def inverse_exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    alpha = x[..., 0:1] / radius  # dot(x, mu0, keepdim=True) / R^2 .. <x, mu0> = x[0] * R
    coef = torch.acos(torch.clamp(alpha, min=-1., max=1.)) / sqrt(1. - alpha**2)
    diff = torch.cat((x[..., 0:1] - alpha * radius, x[..., 1:]), dim=-1)  # y - alpha*mu0 = (y[0]-alpha(-R); y[1:])
    return coef * diff


def sample_projection_mu0(x: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    x_expanded = expand_proj_dims(x)
    pt = parallel_transport_mu0(x_expanded, dst=at_point, radius=radius)
    x_proj = exp_map(pt, at_point=at_point, radius=radius)
    return x_proj, (pt, x)


def inverse_sample_projection_mu0(x: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tensor]:
    unmapped = inverse_exp_map(x, at_point=at_point, radius=radius)
    unpt = inverse_parallel_transport_mu0(unmapped, src=at_point, radius=radius)
    return unmapped, unpt[..., 1:]


def spherical_to_projected(x: Tensor, radius: Tensor) -> Tensor:
    return radius * x[..., 1:] / (radius + x[..., 0:1])

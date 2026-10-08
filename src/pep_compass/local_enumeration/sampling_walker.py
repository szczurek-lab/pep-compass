from abc import ABC
import numpy as np
import torch
from scipy.optimize import root_scalar

from pep_compass.models.encoder_decoder.encoder_decoder import EncoderDecoder


class SubRiemannianTangentSpace:

    def __init__(self, U, S, V, horizontal_threshold, device="cpu"):
        self.U = U
        self.S = S
        self.V = V
        self.horizontal_threshold = horizontal_threshold
        self.device = device

        self.horizontal_dim = torch.sum(
            torch.abs(self.S) > self.horizontal_threshold
        ).item()
        self.vertical_dim = torch.sum(
            torch.abs(self.S) <= self.horizontal_threshold
        ).item()

        self.projection_matrix = None

    def _sample_sphere_tensor(self, size):
        normal_np = np.random.normal(loc=0, scale=1, size=(size,))
        normal = torch.tensor(normal_np, device=self.device).to(torch.float32)
        norm = torch.sum(normal**2) ** 0.5
        result = normal / norm
        
        return result

    def sample_horizontal_direction(self):
        # TODO: Is there any chance that values in S are negative?
        S_abs = torch.abs(self.S)

        horizontal_euc_sphere_sample = self._sample_sphere_tensor(self.horizontal_dim)

        S_horizontal = self.S[S_abs > self.horizontal_threshold]
        # Here we have division in which small values of S can cause numerical instability.
        horizontal_riem_sphere_sample = horizontal_euc_sphere_sample / S_horizontal

        horizontal_riem_sphere_sample_with_zeroed_vertical = torch.zeros_like(self.S)
        horizontal_riem_sphere_sample_with_zeroed_vertical[
            S_abs > self.horizontal_threshold
        ] = horizontal_riem_sphere_sample

        # In each row of the following matrix we have a scaled vector. Some of them are zero vectors.
        horizontal_sample = torch.matmul(
            horizontal_riem_sphere_sample_with_zeroed_vertical, self.V.T
        ) * (self.horizontal_dim**0.5)
        # TODO: Should we multiply by dim ** 0.5 or not?

        return horizontal_sample

    def sample_vertical_direction(self):
        S_abs = torch.abs(self.S)

        vertical_euc_sphere_sample = self._sample_sphere_tensor(self.vertical_dim)

        vertical_euc_sphere_sample_with_zeroed_horizontal = torch.zeros_like(self.S)
        vertical_euc_sphere_sample_with_zeroed_horizontal[
            S_abs <= self.horizontal_threshold
        ] = vertical_euc_sphere_sample

        vertical_sample = torch.matmul(
            vertical_euc_sphere_sample_with_zeroed_horizontal, self.V.T
        ) * (self.vertical_dim**0.5)

        return vertical_sample

    def project_ambient_vector_to_horizontal_space(self, ambient_vector):
        if self.projection_matrix is None:
            # Compute jacobian taking to account only horizontal directions
            S_horizontal_with_vertical_zeroed = torch.Tensor(self.S)
            S_horizontal_with_vertical_zeroed[self.S <= self.horizontal_threshold] = 0
            jac_horizontal = torch.matmul(
                torch.matmul(self.U, torch.diag(S_horizontal_with_vertical_zeroed)),
                self.V.T,
            )

            # TODO: It is a projection. Derive this formula once again.
            self.projection_matrix = torch.linalg.pinv(jac_horizontal)

        projected_horizontal_vector = torch.matmul(
            self.projection_matrix, ambient_vector
        )

        return projected_horizontal_vector


class SubRiemannianManifold:

    def __init__(self, encoder_decoder: EncoderDecoder, horizontal_threshold: float):
        self.horizontal_threshold = horizontal_threshold
        self.encoder_decoder = encoder_decoder

    def get_sub_riemannian_tangent_space(
        self, latent_position
    ) -> SubRiemannianTangentSpace:

        # TODO: try to remove this check
        if latent_position.ndim == 1:
            latent_position = latent_position.unsqueeze(0)

        decoder_jacobian = self.encoder_decoder.decoder_jacobian(latent_position)[0]

        U, S, V = torch.linalg.svd(decoder_jacobian, full_matrices=False)

        return SubRiemannianTangentSpace(
            U, S, V, self.horizontal_threshold, device=self.encoder_decoder.device
        )

    def get_ambient_manifold_acceleration(self, latent_position, latent_velocity):
        ambient_manifold_acceleration = self.encoder_decoder.field_derivative(
            latent_position,
            latent_velocity,
        )[0]
        # TODO: This indexing above is unclean. It should be somehow removed.

        return ambient_manifold_acceleration

    def get_ambient_covariant_derivative(self, latent_point, vector1, vector2):
        ambient_covariant_derivative = (
            self.encoder_decoder.get_ambient_covariant_derivative(
                latent_point,
                vector1,
                vector2,
            )[0]
        )

        return ambient_covariant_derivative

    def get_ambient_position(self, latent_position):
        return self.encoder_decoder.decoder_forward(latent_position)

    def get_latent_position(self, ambient_position):
        return self.encoder_decoder.encoder_forward(ambient_position)

class SamplingWalker(ABC):
    """
    Abstract base class for sampling walkers in latent space.

    This class defines the interface for sampling walkers that perform steps in a latent space,
    typically used in sampling algorithms or optimization routines. Subclasses must implement
    the `step` method to define how a single step is taken from a given latent position.
    """
    def step(self, latent_position: torch.Tensor) -> tuple:
        raise NotImplementedError("This method should be overridden by subclasses.")

class SecondOrderRiemannianBrownianEfficientSampling(SamplingWalker):
    # TODO: review the docstring
    """
    Efficient second-order sampler for sub-Riemannian Brownian motion horizontal
    dynamics with optional vertical diffusion.

    The sampler uses a second-order Taylor approximation of the horizontal
    position update
      delta_x = -0.5 * a * dt + v * sqrt(dt)
    where `a` is the horizontal component of manifold acceleration (ambient
    acceleration projected to horizontal space), `v` is a sampled horizontal
    velocity direction, and `dt` is the time step.

    The implementation limits the norm of the horizontal position update to
    `max_horizontal_update_norm` by shrinking the effective sqrt(dt) when needed.

    Parameters
    - manifold: object exposing SubRiemannianManifold interface.
    - time_step: float time discretization step.
    - max_horizontal_update_norm: maximum allowed Euclidean norm of the horizontal position update.
    - vertical_movement: if True, add an isotropic vertical diffusion term sampled from tangent space.
    """

    def __init__(
        self,
        encoder_decoder: EncoderDecoder,
        horizontal_threshold: float = 0.1,
        time_step: float = 0.01,
        max_horizontal_update_norm: float = 0.5,
        vertical_movement: bool = True,
    ):
        self.manifold = SubRiemannianManifold(
            encoder_decoder=encoder_decoder, horizontal_threshold=horizontal_threshold
        )
        self.time_step = time_step
        self.max_horizontal_update_norm = max_horizontal_update_norm
        self.vertical_movement = vertical_movement

    def _get_horizontal_manifold_acceleration(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        tangent_space: SubRiemannianTangentSpace,
    ) -> torch.Tensor:
        """
        Compute the horizontal component of the manifold acceleration:
        project the ambient manifold acceleration onto the horizontal subspace.
        """

        ambient_manifold_acceleration = self.manifold.get_ambient_manifold_acceleration(
            position,
            velocity,
        )

        horizontal_manifold_acceleration = (
            tangent_space.project_ambient_vector_to_horizontal_space(
                ambient_manifold_acceleration
            )
        )

        return horizontal_manifold_acceleration

    @staticmethod
    def _vector_quadratic_function(
        A: np.ndarray, B: np.ndarray, t: float
    ) -> np.ndarray:
        """
        Evaluate A * t^2 + B * t for vector-valued A, B given scalar t.
        """
        return A * (t**2) + B * t

    def _compute_sqrt_time_that_satisfies_bound(
        self,
        A_tensor: torch.Tensor,
        B_tensor: torch.Tensor,
        max_norm: float,
        default_sqrt_t: float,
    ) -> float:
        """
        If the vector function f(t) = A * t^2 + B * t has norm <= max_norm at
        default_sqrt_t, return default_sqrt_t. Otherwise solve for the largest
        t in (0, default_sqrt_t] such that ||f(t)|| = max_norm.

        Inputs A_tensor and B_tensor are torch.Tensors; they will be converted to numpy arrays.
        Returns a scalar sqrt(dt) value (float).
        """
        # Convert to numpy arrays on CPU for scipy root finding
        A_np: np.ndarray = A_tensor.cpu().detach().numpy()
        B_np: np.ndarray = B_tensor.cpu().detach().numpy()

        def root_func(t: float) -> float:
            return float(
                np.linalg.norm(self._vector_quadratic_function(A_np, B_np, t))
                - max_norm
            )

        # If default already below bound, return it
        if root_func(default_sqrt_t) <= 0.0:
            return default_sqrt_t

        adjusted_sqrt_t = root_scalar(root_func, bracket=(0.0, default_sqrt_t))

        return adjusted_sqrt_t.root

    def _get_horizontal_position_update(
        self,
        velocity: torch.Tensor,
        manifold_acceleration: torch.Tensor,
        time_step: float,
    ) -> torch.Tensor:
        """
        Compute the horizontal position update vector using the second-order expansion:
          delta = -0.5 * a * dt + v * sqrt(dt)
        """
        sqrt_dt = float(np.sqrt(time_step))
        return -0.5 * manifold_acceleration * (sqrt_dt**2) + velocity * sqrt_dt

    def _get_adjusted_time_step(
        self,
        horizontal_velocity: torch.Tensor,
        horizontal_acceleration: torch.Tensor,
    ) -> float:
        """
        Return an adjusted time step dt (<= configured time_step) such that the
        Euclidean norm of the horizontal position update does not exceed
        max_horizontal_update_norm.

        The function checks the current time_step; if the horizontal update would
        exceed the limit, it finds a reduced sqrt(dt) that enforces the bound and
        returns its square.
        """

        horizontal_position_update = self._get_horizontal_position_update(
            horizontal_velocity, horizontal_acceleration, self.time_step
        )
        horizontal_position_update_norm = float(
            torch.linalg.norm(horizontal_position_update)
        )

        if horizontal_position_update_norm <= self.max_horizontal_update_norm:
            return self.time_step

        # The quadratic form in variable t = sqrt(dt) is:
        # f(t) = (-0.5 * a) * t^2 + v * t
        A = -0.5 * horizontal_acceleration
        B = horizontal_velocity
        default_sqrt_t = float(np.sqrt(self.time_step))

        adjusted_sqrt_t = self._compute_sqrt_time_that_satisfies_bound(
            A, B, self.max_horizontal_update_norm, default_sqrt_t
        )
        adjusted_dt = min(adjusted_sqrt_t**2, self.time_step)
        return float(adjusted_dt)

    def step(self, latent_position: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Take a single sampling step from `latent_position` and return a tuple
        (new_latent_position, step_info).

        Parameters
        - latent_position: torch.Tensor representing the current latent point.

        Returns
        - new_latent_position: torch.Tensor
        - adjusted_time_step: float
        """
        tangent_space = self.manifold.get_sub_riemannian_tangent_space(latent_position)

        horizontal_velocity = tangent_space.sample_horizontal_direction()
        horizontal_manifold_acceleration = self._get_horizontal_manifold_acceleration(
            latent_position,
            horizontal_velocity,
            tangent_space,
        )

        adjusted_time_step = self._get_adjusted_time_step(
            horizontal_velocity, horizontal_manifold_acceleration
        )

        position_update = self._get_horizontal_position_update(
            horizontal_velocity,
            horizontal_manifold_acceleration,
            adjusted_time_step,
        )

        if self.vertical_movement:
            random_vertical_direction = tangent_space.sample_vertical_direction()
            vertical_update = random_vertical_direction * np.sqrt(adjusted_time_step)
            position_update += vertical_update

        new_latent_position = latent_position + position_update

        step_info = {
            "adjusted_time_step": adjusted_time_step,
            "S": tangent_space.S,
            "U": tangent_space.U,
        }

        return new_latent_position, step_info

class SORBESWithoutManifoldAcceleration(SecondOrderRiemannianBrownianEfficientSampling):
    def __init__(
        self,
        manifold: SubRiemannianManifold,
        spatial_step: float,
        max_horizontal_update_norm: float,
        max_velocity_update_norm: float,
        vertical_movement: bool,
    ):
        super().__init__(
            manifold,
            spatial_step,
            max_horizontal_update_norm,
            max_velocity_update_norm,
            vertical_movement,
        )

    def _get_horizontal_manifold_acceleration(self, position, velocity, tangent_space):
        return torch.zeros_like(position)

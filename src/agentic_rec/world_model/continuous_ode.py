from __future__ import annotations

from agentic_rec.core.linalg import Matrix, Vector, matvec, vector_add, vector_scale
from agentic_rec.world_model.krylov import krylov_expm_action


class LinearWorldModel:
    """
    Minimal continuous-time world model:

        du/dt = A u + B a

    The control integral is approximated with a midpoint rule so we can keep the
    runtime demo zero-dependency. In the PyTorch training path, you can replace
    this with a more exact `matrix_exp`-based implementation.
    """

    def __init__(
        self,
        transition: Matrix,
        control: Matrix,
        delta_t: float = 1.0,
        krylov_steps: int = 4,
    ) -> None:
        self.transition = transition
        self.control = control
        self.delta_t = delta_t
        self.krylov_steps = krylov_steps

    def drift(self, user_state: Vector, action: Vector) -> Vector:
        transition_part = matvec(self.transition, user_state)
        control_part = matvec(self.control, action)
        return vector_add(transition_part, control_part)

    def step_euler(self, user_state: Vector, action: Vector) -> Vector:
        return vector_add(user_state, vector_scale(self.drift(user_state, action), self.delta_t))

    def step_krylov(self, user_state: Vector, action: Vector) -> Vector:
        transition_term = krylov_expm_action(
            self.transition,
            user_state,
            time_step=self.delta_t,
            steps=self.krylov_steps,
        )
        control_direction = matvec(self.control, action)
        midpoint_integral = krylov_expm_action(
            self.transition,
            control_direction,
            time_step=0.5 * self.delta_t,
            steps=self.krylov_steps,
        )
        return vector_add(transition_term, vector_scale(midpoint_integral, self.delta_t))

    def rollout(self, user_state: Vector, actions: list[Vector], method: str = "krylov") -> list[Vector]:
        states = [user_state]
        current = user_state
        for action in actions:
            if method == "euler":
                current = self.step_euler(current, action)
            else:
                current = self.step_krylov(current, action)
            states.append(current)
        return states

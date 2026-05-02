from __future__ import annotations

from collections.abc import Callable

from agentic_rec.core.linalg import (
    Matrix,
    Vector,
    dot,
    expm_taylor,
    linear_combination,
    matvec,
    matrix_scale,
    norm,
    vector_scale,
    vector_sub,
    zeros,
    zeros_matrix,
)


LinearOperator = Matrix | Callable[[Vector], Vector]


def _apply(operator: LinearOperator, vector: Vector) -> Vector:
    if callable(operator):
        return operator(vector)
    return matvec(operator, vector)


def arnoldi_iteration(
    operator: LinearOperator,
    vector: Vector,
    steps: int,
    eps: float = 1e-10,
) -> tuple[list[Vector], Matrix, int, float]:
    """Build a Krylov basis. For symmetric operators this reduces to Lanczos."""
    beta = norm(vector)
    if beta <= eps:
        return [], zeros_matrix(1, 1), 0, 0.0

    basis: list[Vector] = [vector_scale(vector, 1.0 / beta)]
    hessenberg = zeros_matrix(steps + 1, steps)
    actual_steps = 0

    for column in range(steps):
        work = _apply(operator, basis[column])
        for row in range(column + 1):
            projection = dot(basis[row], work)
            hessenberg[row][column] = projection
            work = vector_sub(work, vector_scale(basis[row], projection))

        residual = norm(work)
        actual_steps = column + 1
        if residual <= eps or actual_steps == steps:
            break

        hessenberg[column + 1][column] = residual
        basis.append(vector_scale(work, 1.0 / residual))

    return basis, hessenberg, actual_steps, beta


def krylov_expm_action(
    operator: LinearOperator,
    vector: Vector,
    time_step: float,
    steps: int = 6,
) -> Vector:
    """Approximate exp(tA)v with a small Krylov subspace."""
    basis, hessenberg, actual_steps, beta = arnoldi_iteration(operator, vector, steps)
    if actual_steps == 0:
        return zeros(len(vector))

    reduced = [row[:actual_steps] for row in hessenberg[:actual_steps]]
    exp_reduced = expm_taylor(matrix_scale(reduced, time_step))
    e1 = [1.0] + [0.0 for _ in range(actual_steps - 1)]
    weights = vector_scale(matvec(exp_reduced, e1), beta)
    return linear_combination(basis[:actual_steps], weights)

from __future__ import annotations

import math


Vector = list[float]
Matrix = list[list[float]]


def zeros(size: int) -> Vector:
    return [0.0 for _ in range(size)]


def zeros_matrix(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def identity(size: int) -> Matrix:
    out = zeros_matrix(size, size)
    for index in range(size):
        out[index][index] = 1.0
    return out


def vector_add(left: Vector, right: Vector) -> Vector:
    return [l_val + r_val for l_val, r_val in zip(left, right)]


def vector_sub(left: Vector, right: Vector) -> Vector:
    return [l_val - r_val for l_val, r_val in zip(left, right)]


def vector_scale(vector: Vector, scalar: float) -> Vector:
    return [scalar * value for value in vector]


def dot(left: Vector, right: Vector) -> float:
    return sum(l_val * r_val for l_val, r_val in zip(left, right))


def norm(vector: Vector) -> float:
    return math.sqrt(dot(vector, vector))


def normalize(vector: Vector, eps: float = 1e-12) -> Vector:
    size = norm(vector)
    if size <= eps:
        return zeros(len(vector))
    return vector_scale(vector, 1.0 / size)


def cosine_similarity(left: Vector, right: Vector, eps: float = 1e-12) -> float:
    left_norm = norm(left)
    right_norm = norm(right)
    if left_norm <= eps or right_norm <= eps:
        return 0.0
    return dot(left, right) / (left_norm * right_norm)


def matvec(matrix: Matrix, vector: Vector) -> Vector:
    return [sum(row[col] * vector[col] for col in range(len(vector))) for row in matrix]


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    return [
        [l_val + r_val for l_val, r_val in zip(l_row, r_row)]
        for l_row, r_row in zip(left, right)
    ]


def matrix_scale(matrix: Matrix, scalar: float) -> Matrix:
    return [[scalar * value for value in row] for row in matrix]


def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    rows = len(left)
    cols = len(right[0])
    inner = len(right)
    out = zeros_matrix(rows, cols)
    for row_index in range(rows):
        for inner_index in range(inner):
            left_value = left[row_index][inner_index]
            if left_value == 0.0:
                continue
            for col_index in range(cols):
                out[row_index][col_index] += left_value * right[inner_index][col_index]
    return out


def expm_taylor(matrix: Matrix, terms: int = 18) -> Matrix:
    size = len(matrix)
    result = identity(size)
    current_power = identity(size)
    factorial = 1.0
    for term_index in range(1, terms):
        current_power = matrix_multiply(current_power, matrix)
        factorial *= term_index
        result = matrix_add(result, matrix_scale(current_power, 1.0 / factorial))
    return result


def linear_combination(vectors: list[Vector], weights: Vector) -> Vector:
    if not vectors:
        return []
    out = zeros(len(vectors[0]))
    for weight, vector in zip(weights, vectors):
        out = vector_add(out, vector_scale(vector, weight))
    return out

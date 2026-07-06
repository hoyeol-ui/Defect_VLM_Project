def compute_priority_score(
    consistency: float,
    groundedness: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    groundedness_max: float = 2.0
) -> float:
    """
    Compute acquisition priority score.

    Formula:
        P(x) = alpha * (1 - C(x)) + beta * (1 - G(x) / G_max)

    Where:
        C(x): semantic consistency score
        G(x): groundedness score
        alpha: weight for semantic uncertainty
        beta: weight for groundedness uncertainty

    References:
        - CALD uses consistency as an active learning signal for object detection.
        - EDL-HUA and related detection AL methods motivate uncertainty aggregation.
        - This implementation adapts weighted aggregation for VLM semantic uncertainty
          and label-free pseudo groundedness.
    """
    semantic_uncertainty = 1.0 - float(consistency)
    groundedness_uncertainty = 1.0 - (float(groundedness) / groundedness_max)

    return round(
        alpha * semantic_uncertainty + beta * groundedness_uncertainty,
        6
    )


def compute_aulc(budgets, maps):
    """
    Compute Area Under Learning Curve using trapezoidal rule.
    """
    if len(budgets) != len(maps):
        raise ValueError("budgets and maps must have same length")

    area = 0.0
    for i in range(1, len(budgets)):
        width = budgets[i] - budgets[i - 1]
        height = (maps[i] + maps[i - 1]) / 2.0
        area += width * height

    return round(area, 6)
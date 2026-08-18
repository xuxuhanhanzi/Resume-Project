"""Final-answer reward."""

from forgemm.reasoning.normalizer import answers_match


def answer_reward(prediction: str, reference: str) -> float:
    """Return bounded exact/relaxed answer correctness."""

    return 1.0 if answers_match(prediction, reference) else 0.0

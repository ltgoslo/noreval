from evaluate import load


def bertscore_f1(references, predictions, **kwargs):
    """Computes the F1 score of the BERTScore metric.

    Args:
        references: A list of reference strings.
        predictions: A list of predicted strings.
        **kwargs: Additional keyword arguments.

    Returns:
        The F1 score of the BERTScore metric.
    """
    bertscore = load("bertscore")
    return bertscore.compute(predictions=predictions, references=references, **kwargs)[
        "f1"
    ][0]

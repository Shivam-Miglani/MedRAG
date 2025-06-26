from ragas import evaluate, metrics

def auto_eval(answer: str, contexts: list[str]):
    result = evaluate(
        answer=answer,
        contexts=contexts,
        metrics=[metrics.faithfulness, metrics.answer_relevancy]
    )
    return {k: float(v) for k, v in result.items()}

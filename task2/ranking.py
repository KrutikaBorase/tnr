"""Summarize gold-passage rankings from saved retrieval results."""


def retrieval_ranking(details):
    ranks = []
    for question in details:
        gold = " ".join(question["gold_passage"].split()).casefold()
        rank = next(
            (
                index
                for index, chunk in enumerate(question["top5"], 1)
                if gold in " ".join(chunk["text"].split()).casefold()
            ),
            None,
        )
        ranks.append(rank)
    if not ranks:
        raise ValueError("Retrieval ranking requires at least one question")
    return sum(rank == 1 for rank in ranks), sum(
        1 / rank for rank in ranks if rank is not None
    ) / len(ranks)

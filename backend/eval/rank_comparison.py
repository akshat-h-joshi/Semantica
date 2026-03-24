def build_rank_map(results):
    """
    results: output of recommend_indices
    returns: dict[index -> rank]
    """
    return {
        item["index"]: rank + 1
        for rank, item in enumerate(results)
    }


def compare_ranks(base_results, other_results):
    base_ranks = build_rank_map(base_results)
    other_ranks = build_rank_map(other_results)

    comparison = {}

    for idx, base_rank in base_ranks.items():
        other_rank = other_ranks.get(idx)

        if other_rank is None:
            comparison[idx] = {
                "base_rank": base_rank,
                "other_rank": None,
                "delta": None,
                "status": "missing"
            }
        else:
            delta = other_rank - base_rank
            comparison[idx] = {
                "base_rank": base_rank,
                "other_rank": other_rank,
                "delta": delta,
                "status": "present"
            }

    return comparison
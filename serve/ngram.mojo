"""Bounded prompt lookup proposals; target verification decides acceptance."""


def ngram_propose(history: List[Int], limit: Int) -> List[Int]:
    var n = len(history)
    # Longest suffix first, latest completed occurrence first. Proposals only
    # read committed history, never rejected tokens left in device storage.
    for width in range(min(8, n - 1), 0, -1):
        for start in range(n - width - 1, max(-1, n - 4096), -1):
            var same = True
            for j in range(width):
                if history[start + j] != history[n - width + j]:
                    same = False
                    break
            if same:
                var result = List[Int]()
                for j in range(min(limit, n - start - width)):
                    result.append(history[start + width + j])
                return result^
    return List[Int]()

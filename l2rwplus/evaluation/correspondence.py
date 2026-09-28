"""Evaluation-only correspondence statistics; never used to infer matches."""

from collections import Counter


def audit(groups, truth):
    occurrences = Counter(p for d in truth for p in d.values())
    total_true = sum(n * (n - 1) // 2 for n in occurrences.values())
    correct = accepted = 0
    matched = set()
    for group in groups:
        if len(group) > 1:
            matched.update(group)
        for i, (a, l) in enumerate(group):
            for b, k in group[i + 1 :]:
                accepted += 1
                correct += truth[a][l] == truth[b][k]
    unique = {(c, l) for c, d in enumerate(truth) for l, p in d.items() if occurrences[p] == 1}
    return dict(
        accepted_pairs=accepted,
        correct_pairs=correct,
        precision=correct / accepted if accepted else None,
        pair_recall=correct / total_true if total_true else None,
        unique_identity_false_acceptance=len(unique & matched) / len(unique) if unique else None,
        true_pairs=total_true,
        unique_identities=len(unique),
    )

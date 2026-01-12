"""
@Date  : 2022/12/18
@Time  : 15:18
@Author: Ziyang Huang
@Email : huangzy0312@gmail.com
"""
import torch


def decimal_to_bits(x, bits):
    """ expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1 """
    device = x.device

    mask = 2 ** torch.arange(bits - 1, -1, -1, device=device)
    x = x.unsqueeze(dim=-1)
    # x = rearrange(x, 'b l -> b l 1')

    bits = ((x & mask) != 0).float()
    bits = bits * 2 - 1
    return bits


def bits_to_decimal(x, bits):
    """ expects bits from -1 to 1, outputs image tensor from 0 to 1 """
    device = x.device

    x = (x > 0).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=device, dtype=torch.int32)

    # dec = reduce(x * mask, 'b l d -> b l', 'sum')
    dec = (x * mask).sum(dim=-1)
    # [bsz, len]
    return dec

# utils/cky_projection.py
NEG_INF = -1e9


def extract_span_scores(labels, relation_ids):
    """
    A simple heuristic to produce span-relation scores from a label sequence.

    labels: 1D LongTensor [seq_len] - predicted label ids (int)
    relation_ids: iterable of ints - the IDs corresponding to compound relations
    returns: dict mapping (i,j,r) -> score (float)
    """
    N = labels.size(0)
    scores = {}

    # Heuristic: span score for relation r = fraction of tokens inside span that predict r
    for i in range(N):
        for j in range(i + 1, N):
            span = labels[i:j + 1]
            for r in relation_ids:
                # compute fraction of tokens equal to r
                score = (span == r).float().mean().item()
                scores[(i, j, r)] = score

    return scores


def cky_decode(scores, N, relation_ids):
    """
    Standard CKY-like DP to pick a binary parse over tokens [0..N-1].
    dp[i][j] = best score for span [i..j]
    back[i][j] = (split_k, relation_id)
    """
    dp = [[NEG_INF] * N for _ in range(N)]
    back = [[None] * N for _ in range(N)]

    for i in range(N):
        dp[i][i] = 0.0

    for length in range(2, N + 1):
        for i in range(0, N - length + 1):
            j = i + length - 1
            best_val = NEG_INF
            best_choice = None
            for k in range(i, j):
                left = dp[i][k]
                right = dp[k + 1][j]
                for r in relation_ids:
                    s = scores.get((i, j, r), NEG_INF)
                    val = left + right + s
                    if val > best_val:
                        best_val = val
                        best_choice = (k, r)
            dp[i][j] = best_val
            back[i][j] = best_choice

    return back


def recover_spans(back, i, j, spans):
    """
    Recursively recover spans from back pointers.
    Appends (i,j,relation_id) to spans.
    """
    if i >= j:
        return
    choice = back[i][j]
    if choice is None:
        return
    k, r = choice
    spans.append((i, j, r))
    recover_spans(back, i, k, spans)
    recover_spans(back, k + 1, j, spans)


def project_labels_with_cky(labels, label_set):
    """
    Project a single label sequence (1D LongTensor) to the nearest valid binary-tree
    labeling according to CKY DP and the given label_set.

    labels: LongTensor [seq_len] containing predicted label ids
    label_set: object with attributes:
        - compound_relation_ids: iterable of int (relation ids to consider)
        - no_rel_id: int id for "No_rel"
    Returns:
        projected_labels: LongTensor [seq_len]
    """
    N = labels.size(0)
    if N <= 1:
        return labels.clone()

    # Build span scores
    scores = extract_span_scores(labels, label_set.compound_relation_ids)

    # CKY decode
    back = cky_decode(scores, N, label_set.compound_relation_ids)

    # Recover spans
    spans = []
    recover_spans(back, 0, N - 1, spans)

    # Build projected label sequence: default to NO_REL, then set spans to relation ids
    projected = torch.full_like(labels, fill_value=label_set.no_rel_id)
    for i, j, r in spans:
        # Set every token in span to the relation id (you can choose a head-based encoding if desired)
        projected[i:j + 1] = r

    return projected


if __name__ == '__main__':
    x = torch.randint(0, 8, [8, 12, 12])
    bits = decimal_to_bits(x, 3)
    decimal = bits_to_decimal(bits * 3, 3)
    print(decimal)

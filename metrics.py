from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def tokenize(text):
    """Simple whitespace tokenization + text cleaning."""
    text = text.replace('<|endoftext|>', '')
    return text.split()


def compute_self_bleu(samples, n=4):
    """
    Compute average self‑BLEU score: for each sentence, BLEU against all others as references.
    Lower score indicates higher diversity.
    """
    tokenized_samples = [tokenize(s) for s in samples]
    smoothie = SmoothingFunction().method4
    total_bleu = 0.0
    count = 0

    for i, hyp in enumerate(tokenized_samples):
        # References are all other samples
        refs = [tokenized_samples[j] for j in range(len(samples)) if j != i]
        if not refs:
            continue
        bleu = sentence_bleu(
            refs,                          # list of reference token lists
            hyp,                            # hypothesis token list
            weights=tuple(1./n for _ in range(n)),
            smoothing_function=smoothie
        )
        total_bleu += bleu
        count += 1

    return total_bleu / count if count > 0 else 0.0


def compute_distinct_n(samples, n=2):
    """
    Compute Distinct‑n score (proportion of unique n‑grams among all generated tokens).
    Higher distinct‑n indicates more diversity.
    """
    tokenized_samples = [tokenize(s) for s in samples]
    all_tokens = [token for tokens in tokenized_samples for token in tokens]
    if len(all_tokens) < n:
        return 0.0

    ngrams = zip(*[all_tokens[i:] for i in range(n)])
    ngram_list = list(ngrams)
    unique_ngrams = set(ngram_list)
    return len(unique_ngrams) / len(ngram_list) if ngram_list else 0.0

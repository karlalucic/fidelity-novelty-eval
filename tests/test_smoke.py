"""Smoke tests for the pure-python core: no network, no models.

Run with ``pytest tests/`` from senra-eval/.
"""
import math

import numpy as np


def test_clean_text_strips_header_and_timestamps():
    import vet
    raw = (
        "This transcript was generated automatically. Its accuracy may vary.\n"
        "Ads/promotional sections removed.\n"
        "Source: https://example.com\n"
        "Title: Demo\n"
        "\n"
        "2:53\n"
        "So this is the actual spoken content here.\n"
        "So this is the actual spoken content here.\n"   # duplicate line
    )
    out = vet.clean_text(raw)
    assert "actual spoken content" in out
    assert "transcript was generated" not in out.lower()
    assert "2:53" not in out
    assert out.lower().count("actual spoken content") == 1


def test_ngram_jaccard_identity_and_disjoint():
    import gate
    s = "the quick brown fox jumps over the lazy dog and runs away fast today"
    assert gate.ngram_jaccard(s, s, 5) == 1.0
    assert gate.ngram_jaccard(s, "completely unrelated tokens nothing shared here at all", 5) == 0.0


def test_copy_rate_flags_verbatim_copy():
    import gate
    cfg = {"gate": {"ngram_orders": [5], "jaccard_threshold": 0.15, "window_words": 5}}
    chunk = "i read the biography of the founder and i was completely obsessed with the story"
    # verbatim copy -> jaccard 1.0, flagged
    hit = gate.gate_output(chunk, [chunk], [chunk], cfg)
    assert hit["copy_rate"] == 1.0
    assert hit["copy_flag"] is True
    miss = gate.gate_output("an entirely different sentence about unrelated matters entirely",
                            [chunk], [chunk], cfg)
    assert miss["copy_flag"] is False


def test_distinct_n_bounds():
    import novelty
    assert novelty.distinct_n("a a a a a", [1])["distinct_1"] < 0.5    # repetitive
    assert novelty.distinct_n("a b c d e f g", [1])["distinct_1"] == 1.0  # all unique


def test_dual_divergence_with_fake_embeddings():
    import novelty
    out = np.array([1.0, 0.0, 0.0])
    near = np.array([[1.0, 0.0, 0.0]])   # identical chunk -> low archive divergence
    far = np.array([[0.0, 1.0, 0.0]])    # orthogonal chunk -> high divergence
    d_copy = novelty.dual_divergence(out, near, near[0])
    d_novel = novelty.dual_divergence(out, far, far[0])
    assert d_copy["dist_nearest_chunk"] < d_novel["dist_nearest_chunk"]
    assert d_copy["dist_c1_centroid"] < d_novel["dist_c1_centroid"]
    assert d_copy["dual"] < d_novel["dual"]


def test_burrows_delta_finite_and_ranks_register():
    import fidelity
    cfg = {"fidelity": {"burrows": {"n_function_words": 80}}}
    target = [("i am so obsessed with this and i love it so much it is just so good i mean "
               "i cannot believe it and i keep reading it again and again ") * 6,
              ("this is the thing i love about him he just does it again and again and i am "
               "obsessed it is so good i mean really it is just incredible to me ") * 6]
    floor = [("the quarterly financial results indicate a substantial increase in revenue "
              "attributable to the operational efficiencies realized during the period ") * 6,
             ("pursuant to the aforementioned regulatory framework the committee shall convene "
              "to deliberate upon the proposed amendments to the existing provisions ") * 6]
    target_like = ("i am obsessed with this and i love it so good i mean it is just incredible "
                   "and i keep doing it again and again so good ") * 4
    d_tt = fidelity.burrows_delta(target_like, target, floor, cfg)
    d_tf = fidelity.burrows_delta(target_like, floor, target, cfg)
    assert math.isfinite(d_tt) and math.isfinite(d_tf)
    assert d_tt <= d_tf   # target-like text no further from target than from floor


def test_chunk_archive_shapes():
    import corpus
    text = " ".join(f"word{i}" for i in range(500))
    chunks = corpus.chunk_archive([text], {"retrieval": {"chunk_words": 100, "chunk_overlap_words": 20}})
    assert len(chunks) >= 4
    assert all(set(c.keys()) == {"id", "text"} for c in chunks)


def test_extract_guest_turns_both_label_formats():
    import corpus
    interview = (
        "Title: Demo\n"
        "David Senra: This is the host talking now.\n"
        "James Dyson: This is the guest speaking here.\n"
        "[David Senra] Host again in bracket form.\n"
        "[James Dyson] Guest again in bracket form.\n"
    )
    guest = corpus.extract_guest_turns(interview, "David Senra")
    assert "guest speaking" in guest and "Guest again" in guest
    assert "host talking" not in guest and "Host again" not in guest

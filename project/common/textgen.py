"""
결정적(deterministic) 합성 텍스트 생성 + 토큰 길이 제어.

왜 합성 텍스트인가:
  - Prefill 비용을 논하려면 프롬프트의 **정확한 토큰 길이**가 필요하다.
  - 5명이 수십 번 재실행해도 완전히 동일한 트레이스여야 비교가 가능하다.
  실제 코퍼스로는 이 두 가지를 동시에 만족시키기 어렵다.

`transformers` 와 모델 토크나이저가 있으면 길이가 정확히 맞춰지고, 없으면
문자수 기반 추정으로 대체된다(GPU 없는 노트북에서도 트레이스 생성 가능).
**최종 제출용 트레이스는 반드시 토크나이저를 지정해 한 번 생성하고 git 에 커밋한다.**
"""

from __future__ import annotations

import functools
import math
import random

_CHARS_PER_TOKEN = 4.2  # 토크나이저 없을 때의 추정치(영문 산문 기준)

_VOCAB = """
system latency throughput kernel batch tensor memory bandwidth cache prefix
scheduler request token attention query key value projection layer residual
normalization embedding decode prefill quantization allocation fragmentation
policy eviction retention admission preemption speculative draft verification
document retrieval chunk passage index similarity ranking corpus citation
inference serving cluster node replica shard pipeline parallel overlap graph
budget deadline percentile distribution arrival departure queue occupancy
transaction ledger settlement inventory logistics forecast variance baseline
protocol interface contract migration rollback checkpoint snapshot replication
""".split()


@functools.lru_cache(maxsize=4)
def _get_tokenizer(model: str | None):
    if not model:
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception:
        return None


def count_tokens(text: str, model: str | None = None) -> int:
    tok = _get_tokenizer(model)
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def make_text(n_tokens: int, seed: int, model: str | None = None) -> str:
    """seed 로 결정되는 약 n_tokens 길이의 의사 텍스트를 만든다.

    서로 다른 seed 는 **첫 토큰부터** 다른 텍스트를 만든다. 이게 중요하다.
    '서로 다른 문서'끼리 우연히 프리픽스를 공유하면, 이 스터디의 모든 캐시
    적중률 수치가 조용히 부풀려진다.
    """
    rng = random.Random(seed)
    words: list[str] = [f"[doc-{seed:06d}]"]  # 첫 토큰부터 분기시키는 마커
    target_chars = int(n_tokens * _CHARS_PER_TOKEN)

    while True:
        sent = [rng.choice(_VOCAB) for _ in range(rng.randint(8, 20))]
        sent[0] = sent[0].capitalize()
        words.append(" ".join(sent) + ".")
        if len(" ".join(words)) >= target_chars:
            break

    text = " ".join(words)
    tok = _get_tokenizer(model)
    if tok is None:
        return text

    # 실제 토크나이저가 있으면 정확히 n_tokens 로 맞춘다
    ids = tok.encode(text, add_special_tokens=False)
    while len(ids) < n_tokens:
        text += " " + " ".join(rng.choice(_VOCAB) for _ in range(16)) + "."
        ids = tok.encode(text, add_special_tokens=False)
    return tok.decode(ids[:n_tokens])


def lognormal_int(rng: random.Random, mean: float, sigma: float,
                  lo: int, hi: int) -> int:
    """롱테일 길이 샘플러.

    실제 요청 길이는 균등분포가 아니다. 그리고 스케줄러를 망가뜨리는 것은 정확히
    그 꼬리(tail)다. 절대 uniform 을 쓰지 말 것.
    """
    mu = math.log(mean) - 0.5 * sigma ** 2
    return max(lo, min(hi, int(math.exp(rng.gauss(mu, sigma)))))


def poisson_arrivals(rng: random.Random, qps: float, n: int,
                     start: float = 0.0) -> list[float]:
    """Open-loop 포아송 도착 시각. 요청 간격은 지수분포 Exponential(1/qps)."""
    t, out = start, []
    for _ in range(n):
        t += -math.log(max(rng.random(), 1e-12)) / qps
        out.append(round(t, 4))
    return out

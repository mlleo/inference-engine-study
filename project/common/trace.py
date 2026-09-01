"""
공용 트레이스 스키마.

모든 워크로드 생성기는 이 포맷으로만 출력하고, 재생 하니스(bench/replay.py)만이
이 포맷을 읽는다. 5명이 서로 다른 클라이언트를 쓰면 마지막에 아무것도 비교할 수
없게 된다. **이 파일은 1주차에 확정하고 이후 수정 금지.**

요청 도착 방식은 두 가지다.

  1. 독립 도착 (INDEPENDENT)
     `arrival_time` (실행 시작 후 경과 초)을 가진다. 서버가 밀리든 말든 벽시계
     기준으로 발사된다(open-loop). 실제 트래픽을 모사한다.

  2. 연쇄 도착 (CHAINED)
     `depends_on`(부모 요청 rid)과 `think_time`을 가진다. 부모 요청이 **완료된
     후** think_time 초 뒤에 발사된다. 멀티턴 대화와 에이전트 루프는 반드시 이
     방식이어야 한다. 턴 N이 답을 받기 전에는 턴 N+1이 존재할 수 없기 때문이다.

연쇄 요청의 프롬프트는 재생 시점에 다음과 같이 조립된다.

    부모_프롬프트 + 부모_출력 + suffix

이것이 핵심이다. 공유 프리픽스가 시뮬레이션이 아니라 실제로 만들어지므로,
응답에 담겨 오는 `cached_tokens` 값이 의미를 갖는다. 히스토리를 가짜로 채워
넣으면 W2(에이전트) 과제 전체가 아무것도 측정하지 못하게 된다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Request:
    rid: str
    session_id: str
    turn_idx: int = 0

    # --- 도착 방식: 아래 둘 중 정확히 하나만 설정 ---
    arrival_time: float | None = None   # 독립 도착, t=0 기준 경과 초
    depends_on: str | None = None       # 연쇄 도착, 부모 rid
    think_time: float = 0.0             # 부모 완료 후 대기 시간(초)

    # --- 프롬프트 ---
    prompt: str = ""     # depends_on 이 None 일 때 사용
    suffix: str = ""     # (부모 프롬프트 + 부모 출력) 뒤에 붙일 문자열

    # --- 샘플링 ---
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = True   # 출력 길이를 고정해 측정 재현성 확보
    stop: list[str] = field(default_factory=list)

    # --- 스케줄링 / 분석용 메타데이터 ---
    slo_class: str = "interactive"      # "interactive" | "batch"
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if (self.arrival_time is None) == (self.depends_on is None):
            raise ValueError(
                f"{self.rid}: arrival_time 과 depends_on 중 정확히 하나만 설정해야 함")
        if self.max_new_tokens < 1:
            raise ValueError(f"{self.rid}: max_new_tokens 는 1 이상이어야 함")


@dataclass
class Trace:
    name: str
    requests: list[Request]
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        seen: set[str] = set()
        for r in self.requests:
            r.validate()
            if r.rid in seen:
                raise ValueError(f"rid 중복: {r.rid}")
            seen.add(r.rid)
        for r in self.requests:
            if r.depends_on and r.depends_on not in seen:
                raise ValueError(f"{r.rid}: 존재하지 않는 부모 {r.depends_on}")
        # 순환 참조 검사 (Floyd)
        parent = {r.rid: r.depends_on for r in self.requests}
        for rid in parent:
            slow = fast = rid
            while True:
                fast = parent.get(fast) or ""
                if not fast:
                    break
                fast = parent.get(fast) or ""
                if not fast:
                    break
                slow = parent.get(slow) or ""
                if slow == fast:
                    raise ValueError(f"순환 의존성: {rid}")

    def save(self, path: str) -> None:
        self.validate()
        with open(path, "w") as f:
            f.write(json.dumps({"__meta__": {"name": self.name, **self.meta}},
                               ensure_ascii=False) + "\n")
            for r in self.requests:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    @staticmethod
    def load(path: str) -> "Trace":
        reqs, meta = [], {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "__meta__" in obj:
                    meta = obj["__meta__"]
                    continue
                reqs.append(Request(**obj))
        t = Trace(name=meta.get("name", "unknown"), requests=reqs, meta=meta)
        t.validate()
        return t

    def describe(self) -> str:
        n_chained = sum(1 for r in self.requests if r.depends_on)
        sessions = len({r.session_id for r in self.requests})
        out_toks = sum(r.max_new_tokens for r in self.requests)
        arrivals = [r.arrival_time for r in self.requests if r.arrival_time is not None]
        span = max(arrivals) if arrivals else 0.0
        return (f"트레이스={self.name}  요청수={len(self.requests)}  "
                f"세션수={sessions}  연쇄요청={n_chained}  "
                f"도착구간={span:.1f}s  계획출력토큰={out_toks}")

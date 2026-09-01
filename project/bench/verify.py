"""
정확성 게이트. **속도 향상을 믿기 전에 반드시 먼저 돌린다.**

캐시와 스케줄러 버그는 조용하다. 검색된 컨텍스트를 절반 흘려도, 다른 프리픽스에
속한 KV 블록을 잘못 재사용해도, 수치는 좋아지고 출력만 미묘하게 틀린다.
greedy decode 결과를 비교하면 몇 초 만에 잡힌다.

절차:
    # 1) 수정 전 SGLang 에서 기준 출력 한 번 저장
    python -m bench.replay --trace traces/rag.jsonl --tag ref --keep-output ...
    # 2) 수정 후
    python -m bench.replay --trace traces/rag.jsonl --tag mine --keep-output ...
    # 3) 비교
    python -m bench.verify results/rag__ref.json results/rag__mine.json

판정 기준:
  올바른 수정  -> 일치율 95% 이상, 불일치는 시퀀스 후반에서 발생
                 (배치 구성에 따른 부동소수점 reduction 순서 차이. 정상이다.)
  버그         -> 특정 부분집합에서 **토큰 0번부터** 갈라진다.
                 그 부분집합이 곧 버그 리포트다.

즉, 100% 일치를 요구하지 말고 '일치율'과 '어디서부터 갈라지는가'를 보라.
"""

from __future__ import annotations

import argparse
import json
import sys


def common_prefix_len(a: str, b: str) -> int:
    n, i = min(len(a), len(b)), 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main() -> None:
    ap = argparse.ArgumentParser(description="greedy 출력 정확성 검증")
    ap.add_argument("ref", help="기준(수정 전) 결과 파일")
    ap.add_argument("test", help="검증할(수정 후) 결과 파일")
    ap.add_argument("--min-exact", type=float, default=95.0,
                    help="이 일치율 미만이면 FAIL")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    ref = {r["rid"]: r for r in json.load(open(args.ref))["records"]}
    test = {r["rid"]: r for r in json.load(open(args.test))["records"]}

    shared = [k for k in ref if k in test]
    missing = [k for k in ref if k not in test]
    exact, diverged = 0, []

    for k in shared:
        a, b = ref[k].get("output"), test[k].get("output")
        if a is None or b is None:
            print("에러: 출력이 저장되어 있지 않다. replay 를 --keep-output 으로 재실행할 것.")
            sys.exit(2)
        if a == b:
            exact += 1
        else:
            cp = common_prefix_len(a, b)
            diverged.append((cp / max(len(a), 1), cp, k, a, b))

    rate = 100.0 * exact / max(len(shared), 1)
    print(f"비교 대상   : {len(shared)}건  (test 에 없는 요청: {len(missing)}건)")
    print(f"완전 일치   : {exact}건  ({rate:.1f}%)")

    if diverged:
        diverged.sort()
        early = sum(1 for d in diverged if d[1] < 32)
        print(f"불일치      : {len(diverged)}건  "
              f"(그중 앞 32자 이내에서 갈라진 것 {early}건 <- 이건 부동소수점 노이즈가"
              f" 아니라 버그다)")
        print(f"\n가장 이른 불일치 {min(args.show, len(diverged))}건:")
        for frac, cp, k, a, b in diverged[:args.show]:
            print(f"\n  {k}: 앞 {cp}자 동일 (기준 출력의 {frac:.0%})")
            print(f"    ref : ...{a[max(0,cp-40):cp]}>>>{a[cp:cp+60]}")
            print(f"    test: ...{b[max(0,cp-40):cp]}>>>{b[cp:cp+60]}")

    if rate < args.min_exact or missing:
        print(f"\n=== FAIL (기준 {args.min_exact}%) ===")
        sys.exit(1)
    print("\n=== PASS ===")


if __name__ == "__main__":
    main()

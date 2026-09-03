#!/usr/bin/env python3
import json, glob, os
OUT=glob.glob("/private/tmp/claude-501/*GNN-WRDS/*/tasks/wnwrecu4w.output")[0]
d=json.load(open(OUT))["result"]
secs=d["sections"]; synth=d["synth"]
order=["backbone","board","ownership","analyst","audit","patent","supplychain","mna","secondary"]
secs=sorted(secs,key=lambda s: order.index(s["group"]) if s["group"] in order else 99)

header = """# WRDS 데이터 기반 GNN 연구 가이드

> 생성일 2026-06-02 · WRDS 계정 직접 연결로 검증 · 접근 가능 라이브러리 **220개**
> 실제 WRDS 계정에서 추출한 컬럼 메타데이터·샘플 행을 근거로, 9개 데이터셋 패밀리를 GNN 그래프 관점(노드/엣지/피처/시계열)에서 문서화함.

## 접근 현황 요약 (직접 검증)

| 데이터셋 | 스키마(테이블수) | 규모/커버리지 | 그래프 성격 | 접근 |
|---|---|---|---|---|
| BoardEx 이사회 | `boardex_na`(42) / `boardex`(168) | 북미+유럽+글로벌 | 기업–기업 이사겸임 / 이사–이사 공동이사회 | ✅ Full |
| 기관보유 13F | `tr_13f`(`s34`) | 기관 × 종목 × 분기 | 기관↔종목 이분 + 공동보유 투영 | ✅ Full |
| 애널리스트 IBES | `ibes`(194) | 애널리스트 × 기업 × 추정 | 커버리지 이분망 / 공동커버리지 | ✅ Full |
| 특허 | `wrdsapps_patents`(3) | 미국특허 인용·기업링크 | 특허→특허 인용 유향그래프 | ✅ Full |
| 공급망 | `compseg`(`wrds_seg_customer`) | 기업 → 주요고객 | 공급자→고객 유향그래프 | ✅ Full |
| M&A | `tr_sdc_ma`(6) / `sdc`(68) | 거래·자문사 | 인수자↔대상 / 자문사 공동참여 | ✅ Full |
| **Audit Analytics** | `audit`(396) / `audit_acct_os`(64) / `audit_audit_comp`(59) | 감사·정정·소송·사이버 | 감사인–피감 이분 + **부정회계 라벨** | ✅ Full (두 모듈) |
| 보조망 | `wrdsapps_subsidiary` / `fjc_litigation` / `execcomp` / `trace` | 자회사·소송·임원·채권딜러 | 다양 | ✅ Full |
| 노드피처 백본 | `crsp`(433) / `comp`(293) / `wrdsapps_finratio` | 가격·재무·표준비율 | 모든 그래프의 노드 속성 | ✅ Full |

### ⚠️ 핵심 제약 — 데이터셋 간 링크
- **CCM (CRSP-Compustat Merged, `crsp_a_ccm`) 링크 테이블 접근 거부** → `gvkey`↔`permno` 공식 브리지 없음.
- **확인된 우회 경로:** `wrdsapps_link_crsp_ibes.ibcrsphist`(IBES↔CRSP: ticker↔permno↔ncusip), `wrdsapps_link_crsp_bond.bondcrsp_link`(채권↔CRSP: cusip↔permno↔permco), `wrdsapps_subsidiary.chars`(cik↔gvkey↔cusip↔ticker 브리지). 상세는 **'노드피처 백본'** 절 참조.

---

## 목차

"""
toc="".join(f"{i+1}. {s['title_ko']}\n" for i,s in enumerate(secs))
toc+=f"{len(secs)+1}. 종합: 멀티플렉스 그래프 & 우선순위 연구 어젠다\n"

body=""
for s in secs:
    body+="\n\n---\n\n"+s.get("markdown_section","")

tail="\n\n---\n\n# 종합: 교차-데이터셋 멀티플렉스 그래프 & 우선순위 연구 어젠다\n\n"+synth.get("markdown","")

doc=header+toc+body+tail
open("GNN_WRDS_데이터_가이드.md","w").write(doc)
print("WROTE GNN_WRDS_데이터_가이드.md :", len(doc), "chars,", doc.count("\n"), "lines")

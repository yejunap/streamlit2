# 캐피탈 부품 생산 이익 — 독립 앱 (eve_capital_parts)

본 대시보드에 섞지 않는다 — 별도 창(기본 `:8524`)에서 돈다.
포맷은 사 이 프 대시 보 드 (eve_arb_dashboard) 를 본떴다.

## 판다음 (공식)

    이익/발 = (최저 셀오가 × (1 - 할인) × (1 - 판매세 - 중수수)) × 소출 - 재료비 × (1 + 설치비)

* 재료비는 산다: 같은 허브의 최저 셀오가. 파는 편에만 손비를 얹는다.
* 할인은 필수(기본 10%) — 창가에 매도로 걸어 놓는다에 대한 페널티이다.
* 손비·설치비의 금리는 eveind2(eve_fighter_profit)를 닮았다: 판매세 3.4%, 브로커 1.5%, 설치비(재료비 대비) 3%.

## 돈

    streamlit run apps/eve_capital_parts/app.py
    # 또는 사 이 브 런 터에 부림: serve_local_apps.sh 의 run 8524 한 줄

## 데이터 데움

| 무엇 | 데음 곳 |
|---|---|
| 부품 목록 | ESI 타입 그룹 **873**(기본 25종)·**913**(고급 40종) — 이름·부피는 universe/types |
| 배 목록 | ESI 타입 그룹 **513**(Freighter 6종)·**902**(Jump Freighter 4종)·**463**(T1 바지선)·**543**(T2 바지선)·**464**(천층채굴)·**380**(DST)·**941**(산업지휘선 Orca·Porpoise) — 같은 표에 섞여 돈다 (등급/군 열) |
| 시장 창가 | 리전 주문판 전량 (인접 `esi.py`, ETag·TTL·429 대기가 다 된다) |
| **레씨피** | **`data/recipes.json`** — `tools_build_recipes.py`로 CCP S데에서 채움 |

레씨피를 ESI에서 받지 **못하는** 사정은 이렇다: `universe/blueprints/{id}` 는
존재하지 않는다 — openapi-3.0 전수를 뒤진 데 로 blueprint 엔드는 캐릭터/보유
블루프린트 목록 두 개 밖다. 제조 반응은 CCP **S데**에 있다.

### 레씨피 갱신 — `tools_build_recipes.py`

S데 변환물(Fuzzwork)에서 세 표를 내리여 돌리면 된다:

```
cd /tmp
for t in industryActivity industryActivityMaterials industryActivityProducts; do
  curl -sS -A "Mozilla/5.0" "https://www.fuzzwork.co.uk/dump/latest/csv/$t.csv" -o $t.csv
done
python3 tools_build_recipes.py        # 95/100종(부품 65 + 배 35)을 data/recipes.json 에 적는다
```

### recipes.json 생김새

    {
      "21009": {"materials": [[소재 tid, 수량], ...],
                 "product_qty": 1, "time": 1800, "install_time": 60},
      ...
    }

* key 는 제품 타입 I D — 목록 표에서 확인한다.
* `time` 은 생산 초(발시간), `install_time` 은 생략 만하다.
* 안 박힌 종목은 레씨피 없음으로 뜨기만 하고 계산은 서지 않는다.
* **ORE 배** (예: ORE Strip Miner)는 레씨피는 있어 계산에 뜨나, 정작 BP가 시장
  안 파서 산출된다 — 이 이익은 환상이다. 재료 원가 대비 순이익이 20배 안팎이면
  이것이다.

## 계산 커널

`capital_core.py` 는 네트워크가 전-코 없다 — 받아온 시트만 넣고 돈다. 점검:
`python3 test_capital_core.py`

## 표 세움 — 숫은 숫으로 (`capital_view.py`)

표에 글자를 입히면(1.2B · 3M · 1h 20m) streamlit 은 그 칸을 글자로 대우한다 —
열 머리칸을 눌러 정렬하면 글자 순서대로 센다 (`100M` 이 `9M` 앞에 온다).
그래서 `capital_view.build` 는 숫을 그대로 싣고, 어떤 생김새로 보일 때만 `FORMATS` 에
적는다 — `app.py` 가 그걸 `st.column_config.NumberColumn(format=…)` 로 옮겨 입힌다.

* `소화소요`·`생산시간` 글자도 각각 `소화소요(일)`·`생산시간(h)` 숫으로 바꿨다.
* 빈칸은 `"-"` 글자가 아니라 비어 있다 (글자를 박는 순간 그 열은 글자 열이 된다).
* CSV 도 숫 그대로 나간다 — 엑셀에서 세울 수 있다.
* 점검: `python3 test_capital_view.py`

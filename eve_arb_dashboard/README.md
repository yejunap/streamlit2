# EVE 허브 아비 대시보드 — Jita 4-4 ↔ Amarr · Dodixie · Rens (통합)

세 무역 허브를 Jita와 짝지어 **한 화면에서 견주는** 통합 판입니다.
`eve_arb_dashboard`(Amarr 단독)를 포크했지만 **원본은 건드리지 않았습니다** — 계산 커널은 같고,
허브를 고르고 행에 `쌍` 컬럼을 붙이고 `📊 페어 비교` 탭을 더했습니다.

```bash
pip install -r requirements.txt       # streamlit · pandas · requests
streamlit run app.py --server.port 8581 --server.headless true   # → http://localhost:8581
```

| 앱 | 허브 쌍 | 포트 |
|---|---|---|
| `eve_arb_dashboard` | Jita 4-4 ↔ Amarr VIII | 8577 |
| `eve_fighter_profit` | 전투기 제조 수익 | 8578 |
| `eve_arb_dodixie` | Jita 4-4 ↔ Dodixie IX | 8579 |
| `eve_arb_rens` | Jita 4-4 ↔ Rens VI | 8580 |
| **`eve_arb_multi` (여기)** | **Jita 4-4 ↔ Amarr / Dodixie / Rens** | **8581** |

잠금 암호는 리포에 없다 — 예 값 없이 **시크릿 없으면 안 열린다(fail-closed)**.
잠금 **쌍**: `ARB_ID`(기본 `sl4`) + `ARB_PW`(필수). 로그인할 때 ID랑 암호 둘 다 물는다.
걸면 된다:
* 구름: 대시보드 → Settings → Secrets → `ARB_ID = "sl4"` 줄에 `ARB_PW = "새 비밀번호"` (한 줄이면 끝)
* 로컬: `.streamlit/secrets.toml` 에 `ARB_ID` + `ARB_PW` (`.gitignore`로 무시됨) 또는 환경변수

`5767`은 한때 커밋에 박혀 있던 숫자일 뿐 — 걸기만한 새 비밀번호면 그걸로 치고 받아 쓴다.
시크릿 없이도 여는 창은 없다. 사이드바의 **Jita ↔ 허브 쌍** 에서 필요한 쌍만 고르면 그 주문판만 받는다 —
첫 스캔이 그만큼 줄어든다.

`esi.py`, `arb_core.py`를 고치면 다음 실행 때 자동 반영됩니다 — `freshen.py`가
모듈을 다시 올립니다. 그래도 안 되면 서버를 새로 띄우세요.

사이드바에서 **「주문판 전체 스캔」** 을 누르면 시작됩니다. 첫 실행은 고른 쌍의 리전 전체 주문을 받으므로 수 분이고,
이후에는 ETag 캐시를 때기 때문에 빨라집니다. (페이지 수: The Citadel 397 · Genesis 184 · Sinq Laison 114 · Heimatar 71 — 진행줄은 `esi.region_pages()`가 매 번 셉니다.)

| | 시스템 | 리전 | 스테이션 |
|---|---|---|---|
| Jita 4-4 | `30000142` | `10000002` (The Citadel) | `60003760` (Jita IV - Moon 4) |
| Amarr VIII | `30002187` | `10000043` (Genesis) | `60008494` (Amarr VIII (Oris) - EFA) |
| Dodixie IX | `30002659` | `10000032` (Sinq Laison) | `60011866` (Dodixie IX - Moon 20 - Federation Navy Assembly Plant) |
| Rens VI | `30002510` | `10000030` (Heimatar) | `60004588` (Rens VI - Moon 8 - Brutor Tribe Treasury) |

## 계산하는 4가지 패턴

패턴 이름은 두 번째 허브 자리에 붙습니다 (아마 `X` = 고른 허브).

| 패턴 | 사서 파는 곳 | 운송 |
|---|---|---|
| **A** | X에서 사서 → Jita에 팔기 | O |
| **B** | Jita에서 사서 → X에 팔기 | O |
| **C** | Jita 안에서 사고팔기 | **X (부피 무관)** |
| **D** | X 안에서 사고팔기 | **X (부피 무관)** |

계산은 두 허브의 주문판 → **창 두께로 자른 후보** + **같은 스테이션에서 사자 > 팔기가 뒤집힌 종목**까지 함께 담습니다.

## 표 읽기

| 항목 | 설명 |
|---|---|
| 매집가 / 청산가 | 창가가 아니라, 체결순서로 계산한 평균 체결가 |
| 조마진% / 순이익% | 브로커 수수료 · 판매세를 뺀 뒤의 것 |
| 체결량 | 한계이익이 0 이 되기 전에 채워진 최대 수량 |
| 묶인 자본 | 그 체결량이 드는 돈(ISK) — 예산한도로 걸러라 |
| 총부피 m³ / ISK/m³ | 운송에 필요한 자릿과 자릿 대비 효율 — **가장 중요한 축** |
| 필요 운항 / 1회 운항 이익 | 사이드바의 "1회 적재 가능 부피" 기준 |
| 일평균 거래량 / 소요일 | 하루 평균 거래량과 실제 이탈 가능 기간 |
| ○○호가 유지 h | 그 가격이 시장에서 버틴 시간 — 오래 버틴 게 진짜다 |

C/D 는 한 스테이션에서 **매수호가 > 매도호가**로 뒤집힐 때만 생깁니다. 평소엔 비어있는 게 정상이고, 있다면 거의 오래된 지정가 물량이 하나뿐입니다. 수수료 프리셋을 **`검토 (수수료 0%)`** 로 바꿔 그 차이만 봐라 판단하면 됩니다.

수수료는 기본/실무 프리셋 결과를 겹쳐서 보여주고, 필요하면 직접 입력합니다.
판매 비용은 **판매세**와 **브로커 수수료**로 분리했습니다. 확인해 보니 (EVE Uni Wiki, 2026-08):
브로커 수수료는 **'immediate'가 아닌 지정가를 걸 때만** 부과됩니다 — 즉시판매엔 없습니다.
대신 판매세는 방식(즉시/지정)과 무조건 붙고, 2025-03 패치부터 기본 **7.5%** (Accounting Lv5면 3.4%).
사이드바의 「판매 창을 즉시판매로 가정」이 켜진 상태에서는 판매세만 부과되고,
끄면(지정가 가정) 브로커 수수료가 더해집니다.

## 테스트

```bash
python3 test_arb_core.py              # 계산 커널 — 네트워크 없음
python3 test_app_data.py              # 목데이터 렌더링 — 네트워크 없음
python3 test_app_render.py            # 초기 렌더링 — 네트워크 없음
ESI_LIVE=1 python3 test_app_scan.py   # 실제 ESI 호출 — 느리다
```

## 데이터 소스
* 호가: ESI `GET /markets/{region_id}/orders/` — 리전 전체는 페이지로 받아서 원하지 않는 스테이션을 잘라냄
* 부피: ESI `GET /universe/types/{id}/` — 벌크 엔드포인트가 이 서버용으로 동동 안 되므로 종목별로 부름
* 거래량: ESI `GET /markets/{region}/history/` (30일)

## 전체 아이템 목록 (감사용)
* `python3 build_types.py`가 `data/types.csv`를 만든다 — 시장 통계(`markets/prices`)에 **더하기 두 허브 창에 주문이 있는 전 종목**의 합집합 (현재 19,757행). 통계 목록만 쓰면 Dragonfly/Dromi 같은 미출시(inventory) 아이템이 새므로 창을 반드시 얹는다
* 참고: 이 서버의 `universe/ids`는 본문으로 **따로 배열**을 받아 `inventory_types` 키로 돌려준다 (`{"types":[...]}`는 거부됨)
* 스캐너는 이 목록을 쓰지 **않는다** — 스캔 우주는 이미 두 허브에 주문이 있는 전 종목이다. 목록은 "빼먹은 종목 없는지" 확인하는 감사 용도다

## 통합에서 만진 곳

* `esi.py` — 네 허브를 `HUBS`에 모으고 `SECONDARIES = ["Amarr", "Dodixie", "Rens"]`, 진행 줄용 `region_pages()`
* `arb_core.py` — `patterns_for(hub_b)` 가 짝마다 라벨을 만들고, `analyze_type(..., hub_b=)` 가 행에 `pair`/`hub`를 심는다. `pair_summary(rows)` 가 탭용으로 쌍을 세
* `app.py` — `run_scan(hubs, ...)` 이 Jita를 한 번 받고 짝만큼 받는다 · 후보를 쌍별로 모은다 · `쌍` 컬럼 · `📊 페어 비교` 탭
* `test_app_data.py` — 세 쌍이 한 표에 섞여 뜨는지, 라벨에 허브 이름이 박히는지 본다

원본 `eve_arb_dashboard` 와 계산식을 포개어 본다 — `arb_core.py` 의 계산부는 같다.

```bash
diff -u ../eve_arb_dashboard/arb_core.py arb_core.py | head -40
```

### 페어 비교 우 대
`📊 페어 비교` 는 운송 아비(A/B)만 센다 — C/D는 한 스테이션 내부 일이라 쌍의 수익이 아니다.
합순이익 내림차순이라 맨 위가 그 판의 답이다.


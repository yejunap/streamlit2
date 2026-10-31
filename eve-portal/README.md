# EVE 산업 포털 — Stream런처

기존 8421 포털(server.py)을 있는 그대로 엔진으로 쓰고, 같은 데이터를 streamlit으로도
보는 창구다. 엔진이 SSO·폴링·캐시를 다 담당하고, 화면은 얇게 얹는다.

## 올리기
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
엔진은 8421(또는 config.json의 port)에 자동으로 뜬다. 이미 떠 있으면 그걸 쓴다.

## EVE 개발자 등록 (한 번)
1. https://developers.eveonline.com → 등록할 때 리다이렉트 URI를
   `config.json`의 `redirect_base`와 똑�게 (기본 `http://100.73.78.112:8421/` —
   다른 곳에 올리는 곳이면 여기도 실제 주소로)
2. `config.json`의 `client_id` 를 본인 것으로
3. 탭: `esi-characters`·`esi-wallet.read_character_wallet.v1`·
   `esi-industry.read_character_jobs.v1`·`esi-assets.read_assets.v1` 코퍼 것들 포함
4. **본인 Station Manager 직위**여야 코퍼 자산·작업이 온다

## 파일
| 것 | 뜻 |
|---|---|
| `server.py` | 엔진 — 그대로 돈다 (urllib뿐, 별게 없다) |
| `streamlit_app.py` | 런처 — 이 앱을 위한 얼굴 |
| `apps/` | 원판 계산부(어비·크리스탈·전사기·자본부품) |
| `config.json` | 포트·client_id·허용 캐릭터 |

## 개인것은 안 넣는다
`tokens.json` · `wallet_history.json`은 .gitignore — 토큰은 기계에 남는다.

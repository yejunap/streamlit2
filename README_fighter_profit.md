# EVE Fighter Profit Calculator

T1/T2 전투기 제조 수익 계산기 (Streamlit 앱)

## 실행

```bash
streamlit run app.py
```

- 비밀번호: 시크릿 `ARB_PW` (구름 Settings → Secrets 또는 `.streamlit/secrets.toml`)
- 재료 조달: 시장 팔 호가(lowest sell) 전량 구매 기준
- 시나리오 A: T1 전투기 시장 구매 → T2 제조
- 시나리오 B: T1 전투기 광물으로 자체 제조 → T2 제조
- T1 전투기 12종의 제조 수익(광물 → T1)도 별도 표로 제공
- 허브: Jita/Amarr/Dodixie/Rens/Hek 선택 가능
- 제조 처방 출처: Fuzzwork blueprint API (후즈워크), 시세: EVE ESI

## 참고

- 발명(디크립터/데이터코어) 비용은 제외된 제조 원가 기준입니다.
- ME/TE 연구, 팩토리 보너스, 운송 비용은 반영되지 않았습니다.
- 시트: 10분 캐시(고급 새로고침 버튼으로 초기화)

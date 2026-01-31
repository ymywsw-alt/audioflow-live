# AudioFlow v1 (audioflow-live)

FinishFlow와 동일 스택:
ChatGPT(규칙) → GitHub(코드) → Render(배포) → OpenAI API(레시피 생성) → Render URL(다운로드)

## Endpoints
- GET /        : 최소 UI
- POST /make   : WAV 생성 및 다운로드
- GET /health  : 헬스체크

## Env (Render)
- OPENAI_API_KEY (필수 권장)
- OPENAI_MODEL   (옵션, 기본 gpt-4o-mini)
- MAX_MINUTES    (옵션, 기본 5)

## Notes
- OpenAI는 "음원 파일"을 직접 생성하지 않고, 안전한 파라미터(JSON)만 생성합니다.
- WAV는 서버에서 절차적 합성(멜로디 없음)으로 생성합니다.

from google import genai

# 1. 클라이언트 초기화
# API 키는 환경 변수(GOOGLE_API_KEY 또는 GEMINI_API_KEY)에 설정되어 있다고 가정합니다.
# 직접 입력 시: client = genai.Client(api_key="YOUR_API_KEY")
client = genai.Client(api_key="AIzaSyDirAaBsDDJdKAuaEwFIsSowLnLUcWjfok")

try:
    print("=== 사용 가능한 모델 목록 (google-genai SDK) ===\n")

    # 2. client.models.list()를 통해 모델 목록 가져오기
    # config 파라미터를 통해 페이지 크기 등을 조절할 수 있습니다.
    for model in client.models.list():
        # 3. 모델 정보 출력
        print(f"모델 이름 (Name): {model.name}")
        print(f"표시 이름 (DisplayName): {model.display_name}")

        # 4. 지원 기능 확인 (기존 supported_generation_methods -> supported_actions로 변경됨)
        # 예: 'generateContent'가 있으면 텍스트/멀티모달 생성 가능
        if model.supported_actions:
            print(f"지원 액션: {model.supported_actions}")

        print("-" * 40)

except Exception as e:
    print(f"오류가 발생했습니다: {e}")

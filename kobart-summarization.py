# -*- coding: utf-8 -*-
"""
Created on Fri May  9 00:08:56 2025

@author: 김성령
"""
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')

article = """최근 발생한 SK텔레콤 유심(USIM) 해킹 사고를 악용해 보이스피싱과 스미싱을 시도하는 사례가 발견돼 각별한 주의가 요구된다. 한국인터넷진흥원(KISA)은 8일 보안공지를 통해 'SKT 유심 해킹', '악성앱 감염' 등의 문구를 포함한 문자가 유포되고 있으며 이를 통해 사용자를 속여 악성 앱 설치 및 민감정보 탈취를 시도하는 정황이 확인됐다고 밝혔다. 이번 피싱은 정부기관이나 SK텔레콤를 사칭하며 접근해 가짜 고객센터 번호로 전화하도록 유도한 뒤, 원격 제어 앱 설치를 요구하는 방식으로 이뤄진다. 공격자는 보안 점검, 악성 앱 검사, 피해 구제 등의 명목을 내세워 피해자가 공식 앱스토어에서 원격제어 앱을 직접 설치하도록 유도한다. 해당 앱이 설치되면, 공격자는 이를 통해 사용자의 스마트폰을 원격으로 조작하며 개인정보, 금융정보 등 민감한 정보를 빼내거나 추가 악성 앱을 설치할 수 있다. KISA는 "정부기관이나 SK텔레콤은 원격제어 앱 설치를 요구하지 않는다"고 강조하며 유사한 메시지를 수신한 경우 링크 클릭이나 앱 설치, 전화 연결을 자제하고 즉시 삭제할 것을 당부했다.
스미싱 피해를 막기 위해서는 먼저 문자 수신 시 출처가 불분명한 링크는 클릭하지 말고 즉시 삭제하는 것이 중요하다. 의심되는 웹사이트 주소의 경우, 실제 정상 사이트와 URL이 일치하는지 반드시 확인해야 한다. 아울러 휴대전화번호, 아이디, 비밀번호 등 개인정보는 반드시 신뢰할 수 있는 사이트에서만 입력해야 하며, 인증번호 입력 시에는 모바일 결제로 연계될 수 있으므로 한 번 더 확인이 필요하다. 악성앱 감염 또는 피싱 사이트를 통해 개인정보가 유출된 경우, 피해자의 번호가 도용돼 스미싱 문자 발송에 악용될 수 있다. 스미싱 악성앱에 감염되거나 피싱 사이트에 개인정보를 입력한 경우, 모바일 소액결제 피해로 이어질 수 있다. 피해가 의심될 경우 ▲통신사 고객센터를 통해 모바일 결제 내역을 확인 ▲피해가 확인되면 스미싱 문자 내용을 캡처 ▲통신사 고객센터에 스미싱 피해 신고 및 소액결제 확인서 발급 요청 등의 절차를 따르도록 한다．
"""

input_ids = tokenizer.encode(article, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(
    input_ids,
    max_length=128,
    min_length=30,
    num_beams=4,
    length_penalty=1.5,
    repetition_penalty=2.0,
    no_repeat_ngram_size=3,
    early_stopping=True
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("요약 결과:", summary)


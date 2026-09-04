# HWM_PLDM 원본 대비 추가된 실험 코드 (fork: slcnvly/HWM_PLDM, branch: lang-align).
#
# 목적: level2의 고수준 잠재 행동 l_t(posterior_mu)를, 웨이포인트 구간의 이동을 설명하는
# 규칙 기반 자연어 캡션의 문장 임베딩과 정렬시키는 보조 대조 손실(LangAlign objective)을
# 추가한다. HWM 논문에서 level2 backbone이 identity_encoder라 level1과 표현 공간을
# 공유하는 문제(계층이 시간 스케일로만 구현됨)를 완화하려는 목적.
#
# 이 패키지의 파일은 전부 신규 추가분이며, 원본 objectives(prediction.py, vicreg.py 등)는
# 건드리지 않는다. 원본 파일 중 실제로 수정한 곳은 각 파일에 동일한 태그로 주석 표시했다.

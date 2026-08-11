# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 강지수
- 리뷰어 : 이민서


# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**

루브릭 5개 항목을 모두 충족합니다.

BasicBlock / BottleneckBlock 및 ResNetModel로 네 모델(ResNet-34/50, PlainNet-34/50)을 정상 구현
dummy input (1,3,224,224) forward 시 네 모델 모두 (1, 37) logit 출력 확인
실제 GPU 학습이 실행되어 loss 감소·accuracy 상승이 출력에 남아 있음
동일 split·transform·epoch·optimizer·lr로 4모델 통제 학습
validation accuracy 기준 Ablation 결과표 완성

<img width="1914" height="634" alt="image" src="https://github.com/user-attachments/assets/a3eee617-98b6-46e9-85a6-9035c9c54dff" />
<img width="1844" height="666" alt="image" src="https://github.com/user-attachments/assets/93ddef8f-e371-4a49-a151-53b4e0022a7d" />

    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**

가장 핵심적인 부분은 use_residual 플래그 하나로 ResNet과 PlainNet을 동시에 만들어내는 블록 설계와 그 안의 shortcut 분기입니다. Ablation의 동일 구조에서 잔차만 제거하는 원칙이 바로 이 코드에서 구현되기 때문에 가장 중요하다고 봤습니다.

특히 shortcut 판단 로직이 잘 주석·구조화돼 있어 이해가 쉬웠습니다.

<img width="680" height="828" alt="image" src="https://github.com/user-attachments/assets/39cb3771-beb1-4b33-bf53-8d91638278de" />
<img width="784" height="1066" alt="image" src="https://github.com/user-attachments/assets/0a820a23-986d-45fa-bb30-b8132875ecd4" />

BottleneckBlock의 1×1 → 3×3 → 1×1 각 conv마다 "채널 축소 / 특징 학습 / 채널 4배 확장" 주석이 붙어 있어 expansion 개념도 바로 읽혔습니다.

        
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나 새로운 시도 또는 추가 실험을 수행해봤나요?**

평가 기준 이상의 시도가 많습니다.

CPU 점검 → GPU 본학습 2단계 전략과 함수 재정의(AMP: torch.amp.autocast + GradScaler)로 자원을 아낀 점
재현성·공정성 인프라: stratified split을 JSON으로 저장·재사용하고, assert로 train/val 무결성(중복 없음·누락 없음)까지 검증
run_experiment에서 모델마다 seed 재설정 + DataLoader/모델/옵티마이저 재생성 + gc.collect()/empty_cache()로 GPU 상태 격리

다만 CPU Smoke Test 출력의 Val Loss 3466.2839는 명백한 이상치인데(BN이 배치 4·8 극소 샘플에서 불안정), 이게 왜 그런지 짚고 넘어간 주석이 없습니다. BatchNorm의 누적 통계(running mean/var)가 극소 데이터라 덜 채워져서 튄 값이고, 본학습에서는 정상이라는 문구를 남겼으면 훌륭한 디버깅 기록이 됐을 것 같습니다.

<img width="602" height="840" alt="image" src="https://github.com/user-attachments/assets/04e0f0a5-71d3-49e1-b83c-9e4027ad08f9" />
<img width="724" height="796" alt="image" src="https://github.com/user-attachments/assets/4e19d045-a024-4b0f-bea0-5ac7de294bf9" />

        
- [x]  **4. 회고를 잘 작성했나요?**

결론 및 회고가 실험 결과 / 해석 / 한계 / 회고로 구조화돼 있어 매우 좋습니다. 특히 3 epoch·단일 seed라 최종 수렴 성능이 아니라는 한계를 명시한 점, residual이 gradient 통과 경로를 제공한다는 해석이 결과와 연결된 점이 인상적입니다.

다만 중간중간 텍스트 흐름(```text 블록)은 있지만 전체 파이프라인을 한눈에 보여주는 플로우 그래프는 없습니다. 전체 파이프라인을 보여주는 다이어그램 하나를 추가하면 완벽할 것 같아요.

<img width="1884" height="1028" alt="image" src="https://github.com/user-attachments/assets/bbf404cd-c4c2-4a6d-946d-393ea1bbbf55" />

        
- [x]  **5. 코드가 간결하고 효율적인가요?**

PEP8을 잘 지켰고 모듈화 수준이 높습니다. use_residual 플래그 + RESNET_CONFIGS 딕셔너리 기반 빌더로 네 모델을 중복 없이 생성하고, create_dataloaders / train_one_epoch / validate_one_epoch / fit_model / run_experiment로 학습 파이프라인을 깔끔히 함수화했습니다. 코드 중복이 최소화된 것 같습니다.

<img width="764" height="638" alt="image" src="https://github.com/user-attachments/assets/04f7a9e9-b791-4b41-9e04-48b50a8cdc22" />
<img width="770" height="1138" alt="image" src="https://github.com/user-attachments/assets/66755179-a6df-40dc-a178-7aee4fc2e04f" />


# 회고(참고 링크 및 코드 개선)
```
# 리뷰어 회고
- use_residual 플래그 하나로 ResNet/PlainNet을 한 코드에서 생성하는 설계가 ablation study의 "통제 변인" 취지와 정확히 맞아떨어져 인상 깊었다.
- 재현성(고정 split JSON + assert 검증)과 자원 관리(CPU 점검→GPU 본학, AMP를 통한 메모리 절, 모델 cleanup)를 꼼꼼히 챙긴 점에서 실험 위생이 좋다고 느꼈다.

# 코드 개선 제안
1) 결과 재현성: DataLoader의 worker 시드까지 고정하면 num_workers=2 환경에서 완전 재현이 가능하다.
      def seed_worker(worker_id):
          import numpy as np, random
          s = torch.initial_seed() % 2**32
          np.random.seed(s); random.seed(s)
      # create_dataloaders의 train_loader에 worker_init_fn=seed_worker 추가

2) 시간 비교 공정성: PlainNet-34 epoch1이 420초(다운로드/워밍업 포함)라 total_seconds가 왜곡됨. best/avg epoch time을 별도 지표로 두면 모델 간 시간 비교가 정확해진다.
```

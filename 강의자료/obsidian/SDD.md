# SDD
- Single Shot MultiBox Detector 
- SSD(Single Shot MultiBox Detector)는 컴퓨터 비전 작업을 위한 합성곱 신경망 모델.
- 이미지 내 객체 감지를 위해 다양한 위치에서 여러 경계 상자(앵커 박스)를 예측. 이를 멀티 박스 탐지(Multi-box Detection)라고 함.
- 사전에 정의된 앵커 박스 안에 객체 존재 여부를 예측하여 객체 인식.
- 예측된 경계 상자와 실제 객체 간 IoU를 계산하여 객체 존재 여부 판단. 일정 이상 IoU는 양성(Positive), 이하는 음성(Negative)으로 레이블링, 학습.
- 객체 감지와 분류를 동시에 수행하는 1단계(1-stage) 구조 객체 인식 모델로, 빠르고 정확.
- R-CNN 계열은 영역 제안과 객체 인식 과정이 분리된 2단계(2-stage) 구조.

- 2-stage 모델은 객체 인식 성능이 높지만, 영역 제안 병목 현상으로 추론 속도가 느림.
	- 자율 주행, 보행자 인식 등 실시간 처리 요구 분야에 활용 어려움.
- 1-stage 모델은 영역 제안 없이 한 번에 객체 인식을 수행하는 단일 네트워크 사용.
- 2-stage 대비 상대적으로 처리 속도가 빠름.
![[Pasted image 20250123015911.png]]
- 그림에서처럼 SSD는 2-stage 객체 감지 알고리즘인 Faster R-CNN보다 빠르고 경량화된 모델 구현 가능.
- 1-stage 객체 감지 알고리즘은 이미지 내 객체 존재 가능성이 있는 모든 위치 추론, 객체 위치 및 클래스 예측.
- 1-stage 알고리즘은 한 번의 순전파로 객체 감지 수행, 처리 속도 빠름.
- SSD 특징 추출 모델은 이미지넷 사전 학습 분류 모델에 여러 계층 추가.
- 다양한 크기 특징 맵 추출, 객체 다양한 크기 인식.
- 영역 제안 결과 필요 없고 병목 현상 사라져 빠른 추론 가능.
# 멀티 스케일 특징 맵
- SSD는 다양한 크기 객체 인식을 위해 다양한 크기 특징 맵 사용.
- 특징 추출 모델 앞부분의 특징 맵은 작은 객체 인식, 뒷부분 특징 맵은 큰 객체 인식.
- 계층별 추출된 특징 맵은 각각 합성곱 계층에 입력, 객체 위치, 클래스 정보로 변환.
- 이 과정을 통해 SSD는 다양한 크기와 종횡비 객체 높은 정확도로 감지.
- 그림 9.9는 SSD 출력 계층 시각화.
![[Pasted image 20250123020024.png]]

- 그림 9.9에서 특징 맵은 3차원 텐서 형태.
- 9.1절 'Faster R-CNN' 설명처럼 그리드 방식 적용.
- 입력 이미지는 그리드 분할, 각 그리드 셀은 객체 존재 여부 판단 작은 네트워크.
- 그리드 셀 내 객체 존재 가능성 있는 앵커 박스 예측.
- 그리드 좌표는 원본 이미지 상대적 위치, 고정 크기 그리드 셀로 분할.
- 그리드 벡터는 앵커 박스 조정값, 클래스 분류 점수로 구성.
- 앵커 박스는 다양한 크기, 종횡비. 객체 존재 가능 위치 대표.
- 앵커 박스별 객체 위치, 클래스 예측. 그리드 셀 내 최고 확률 객체 선택, 최종 탐지 결과 도출
# 기본 박스 
- SSD는 Faster R-CNN 앵커 박스와 유사한 기본 박스(Default Box) 사용.
- 차이점은 서로 다른 크기 특징 맵에 적용한다는 것.
- SSD 모델은 38x38, 19x19, 10x10, 5x5, 3x3, 1x1 크기 특징 맵 각 셀마다 기본 박스 생성.
- 기본 박스 크기는 특징 맵 크기에 따라 결정.
- 작은 특징 맵은 큰 기본 박스로 큰 객체 인식. 큰 특징 맵은 작은 기본 박스로 작은 객체 인식.
- 입력 이미지, 특징 맵 크기 고려하여 초기 기본 박스 크기 설정.
- 다양한 스케일 값 적용, 기본 박스 크기 조정. 스케일은 수식 9.12로 계산.

**수식 9.12 기본 박스 스케일 설정**
![[Pasted image 20250123020143.png]]
- `S_min`, `S_max`는 기본 박스 최소, 최대 크기.
- `m`은 사용할 스케일 개수.
- `k`는 1부터 `m`까지 정수, 각 스케일 색인. 즉, 특징 맵 추출 순서.
- `S_min`, `S_max`는 SSD 학습 전 초깃값 할당, 보통 0.2, 0.9, 6 사용.
- 첫 번째 특징 맵 기본 박스 스케일은 0.2, 두 번째는 0.34.
- 기본 박스 스케일 계산 후, 너비, 높이 스케일 계산. 수식 9.13, 9.14 참고.

**수식 9.13 기본 박스 너비 스케일**
![[Pasted image 20250123020300.png]]
- `w_k^a`는 `k`번째 기본 박스 너비 스케일.
- `s_k`는 기본 박스 크기 스케일.
- `a_r`은 기본 박스 종횡비.

**수식 9.14 기본 박스 높이 스케일**
![[Pasted image 20250123020311.png]]
- $h_k$`는 `k`번째 기본 박스 높이 스케일.
- 너비 스케일과 유사하나, 동일 스케일링은 아님.
- 객체 가로, 세로 길이 다르므로, 각각 맞춰 스케일링해야 정확.
- 예: 사람 인식 시 높이 > 너비. 높이 맞춰 스케일링해야 정확.
- SSD는 기본 박스 너비, 높이 따로 스케일링.
- `a_r` 초깃값 설정, 1, 2, 3 등 사용. 예: 1은 정사각형, 2는 가로가 세로 2배. 다양한 비율 사각형 구성, 다양한 크기, 형태 객체 인식.
- 기본 박스 비율 1:1이면 `s_k`인 기본 박스 추가. 첫 특징 맵은 객체 인식 주로 사용. 종횡비 1 추가 박스로 정확한 크기, 종횡비 작은 객체 인식.

# 모델 학습 
- SSD는 Faster R-CNN과 동일하게 박스 분류, 회귀에 손실 함수 계산. 
- SSD는 영역 제안 네트워크가 없어 배경 영역 사전 선별 불가. 출력값 대부분이 배경.
- 모든 출력값 학습 시, 배경-객체 간 클래스 불균형 발생, 학습 방해.
- 해결 위해 어려운 부정 샘플 마이닝(Hard Negative Mining) 기법 사용.
- Hard Negative Mining은 모델이 잘못 예측, 학습이 어려운 부정적 샘플 선택, 추가 학습.
- SSD는 객체(Positive):배경(Negative) = 1:3 비율로 학습.
- 객체 분류 결과 클래스 점수 정렬, 상위 N개 객체, 배경 샘플 1:3 추출.
- 해당 출력값만 학습. 모델이 배경 더 잘 구분, 객체 더 잘 탐지.
# 실습 
- SSD 모델은 SSD300, SSD512, SSD-MobileNet, SSD-ResNet 등이 있음.
- SSD300, SSD512는 입력 이미지 크기 각각 300x300, 512x512.
- SSD300은 작은 객체 인식, SSD512는 큰 객체 인식에 강점.
- SSD-MobileNet, SSD-ResNet은 각각 MobileNet, ResNet 기반 학습 모델.
- SSD-MobileNet은 경량화, SSD-ResNet은 고정확도 모델.
- 이번 실습은 ResNet-34 특징 추출, SSD512 사용.
- SSD512 특징 추출 네트워크는 토치비전 미지원, 계층 추가해 생성.
- SSD512 특징 추출 네트워크 정의 방법.
```python
class SSDBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        layer1 = backbone.layer1
        layer2 = backbone.layer2
        layer3 = backbone.layer3
        layer4 = backbone.layer4

        self.features = nn.Sequential(layer0, layer1, layer2, layer3)
        self.upsampling= nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.extra = nn.ModuleList(
            [
                nn.Sequential(
                    layer4,
                    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=4),
                    nn.ReLU(inplace=True),
                )
            ]
        )


    def forward(self, x):
        x = self.features(x)
        output = [self.upsampling(x)]

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])
```

- SSD는 멀티 스케일 특징 맵 활용, 여러 계층에서 특징 맵 추출, 다양한 크기 객체 탐지.
- 네트워크 분기를 위해 백본 모델 계층 분리.
- 백본 ResNet-34는 입력 줄기(layer0), 4개 스테이지(layer1, layer2, layer3, layer4) 구성.
- 3번째 스테이지에서 분기, 3번째까지 `features`로 정의. 이후 `upsampling` 계층, 추출 특징 맵 차원 수 증가.
- `extra`는 ResNet-34 마지막 계층 연결 계층 블록, 멀티 스케일 특징 맵 추출 계층.
- `extra` 첫 번째 요소는 ResNet 4번째 스테이지 입력, 나머지 계층 연결.
- ReLU `inplace` 매개변수: 입력 텐서 직접 수정 여부 설정. `True` 설정 시, 입력 텐서 직접 수정, 출력 생성 X, 메모리 사용량 감소.
- 순전파 메서드는 계층 순서대로 연결, 순전파 수행.
- 모델 출력은 여러 계층에서 추출된 특징 맵 포함. `extra` 변수 계층 순차 적용, `output` 변수에 누적.
- 생성된 특징 맵은 순서 보장 딕셔너리(`OrderedDict`) 변환, 반환.
- 이 딕셔너리는 클래스 분류, 박스 회귀 네트워크에 전달.
- SSD512 특징 추출 네트워크 정의 후, SSD512 모델 정의. 

```python
import torch
from torchvision.models import resnet34
from torchvision.models.detection import ssd
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator


backbone_base = resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
backbone = SSDBackbone(backbone_base)
anchor_generator = DefaultBoxGenerator(
    aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05, 1.20],
    steps=[8, 16, 32, 64, 100, 300, 512],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ssd.SSD(
    backbone=backbone,
    anchor_generator=anchor_generator,
    size=(512, 512),
    num_classes=3
).to(device)
```

- ResNet-34(이미지넷 데이터셋, 사전 학습) -> SSD 특징 추출 모델로 사용, `SSDBackbone`에 전달.
- `SSDBackbone` 클래스 출력 개수와 동일 구조 갖는 기본 박스 생성자(`DefaultBoxGenerator`) 전달.
- 기본 박스 생성자는 Faster R-CNN 앵커 생성기와 유사. `aspect_ratios`(종횡비), `scales`(비율), `steps`(간격) 입력.
- `aspect_ratios`: 각 위치 생성 기본 박스 가로세로 비율 설정. [2]는 1:2, [2, 3]은 1:2, 1:3 비율 기본 박스 생성.
- `scales`: 생성할 기본 박스 크기 설정. 리스트 내 큰 값은 더 큰 박스 생성.
- `steps`: 기본 박스 다운 샘플링 비율, 직사각형 격자(`Meshgrid`) 생성에 사용.
- 백본 모델은 7개 특징 맵(`upsampling`(1) + `extra`(6)) 반환. 기본 박스 종횡비, 간격 입력 길이 7.
- Faster R-CNN은 출력 채널(`out_channels`) 직접 할당. SSD는 예제 9.12처럼 출력 채널 배열 계산.

```python
def retrieve_out_channels(model, size):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        image = torch.zeros((1, 3, size[1], size[0]), device=device)
        features = model(image)
        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        out_channels = [x.size(1) for x in features.values()]

    model.train()
    return out_channels

print(retrieve_out_channels(backbone, (512, 512)))
```

- `retrieve_out_channels` 함수: 가상 이미지 입력, 특징 맵 추출, 각 계층 출력 채널 수 반환. 백본에 호출해 결과 예상.
- 함수 내: 모델 평가 모드 변경, 가상 이미지 생성, 백본 모델에 전달, 특징만 추출.
- 기본 박스 생성자 기본 박스 종횡비, 간격 변경 시, 백본 모델 구조 변경 필요. 합성곱 계층 커널 크기, 패딩, 간격 등으로 이미지 크기 점진적 축소. 모델 매개변수 예상 구조 맞게 변경.
- 백본, 앵커 박스 설정 완료 후, SSD 클래스 입력. 입력 크기 (512, 512), 클래스 개수 3개로 제한.
- 모델 선언 완료 후, SSD 모델 학습. 데이터셋 선언, 모델 평가 방법은 9.1절 'Faster R-CNN' 예제 9.5 제외, 예제 9.1~9.10 동일. 단, 에폭은 10으로 설정.
- 
SSD에서 반환되는 손실값은 박스 회귀 손실(bbox_regression)과 객체 분류 손실(classification)이 댜 이 손실값은다음과같은딕셔너리 구조로생성된다.

![[Pasted image 20250123021335.png]]SSD 모델은 Faster R— CNN 모델처럼 학습 모드일 때 손실값들을 출력한다. 
두 개의 손실값이 최소가 되는방향으로학습해야하므로모든손실값을더해서 역전파를수행한다. 
학습이 모두 완료됐다면 데스트 데이터세트를 활용해 결팟값과 모델을 평가해 본다. 다음 결과는 모델 추론 시각화 출력값과 모델 평가 결괴~ 보여준다.
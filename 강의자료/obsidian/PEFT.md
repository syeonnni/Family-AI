# PEFT
- Parameter Efficient Fine Tuning
- 대부분의 개인이나 조직들은 GPU를 사용해 모델을 학습시키는데 비용적인 문제로 어려움이 있음 
- 일부 파라미터만 학습하는 PEFT 방법 연구가 활발히 이루어지고 있음 
- 오픈소스 LLM 학습에서 가장 주목받고 많이 활용되는 학습 방법은 모델에 일부 파라미터를 추가하고 그 부분만 학습하는 LoRA(Low-Rank Adaptation) 학습 방식임 
- LoRA에서 더 발전된 모습인 모델의 파라미터를 양자화한 QLoRA(Quantized LoRA)가 있음 
- Adapter Layers
	- 모델의 기존 아키텍처 사이에 작은 신경망 층을 추가 
	- 학습 시 기존 파라미터는 고정되고, Adapter 레이어의 파라미터만 학습하므로 계산량을 줄일 수 있음 
- Prompt Tuning
	- 특정 입력에 대한 응답을 조정함으로써 모델이 새로운 작업을 수행할 수 있도록 함 
	- 이 방법은 모델의 전체 가중치를 변경하지 않고 입력 프롬프트에 해당하는 파라미터만 학습 
	- 프롬프트 파라미터는 모델 전체 파라미터에 비해 휠씬 작기 때문에 학습 시간이 단축 

# LLM의 파라미터 
- 최근 LLM 모델들의 파라미터는 Large라는 이름에 걸맞게 점점 커지고 있음 
- 메타에서 발표한 Llama 3는 8B 및 70B 파라미터를 가지고 있음 
- GPT-3.5는 175B를 시작으로 GPT-4는 약 1조 7천억개의 파라미터가 존재 
## GPU 메모리 요규량 
- 10억개의 파라미터로 구성된 1B 크기의 LLM을 로드하고 학습하는 데 32bit를 예를 들면 아래와 같다 
- 1개의 파라미터를 저장하기 위해서는 4Byte가 필요 
- GPU의 메모리에 모델을 로드하는데 4GB의 VRAM이 필요 
- 학습을 위해서는 모델 파라미터, gradient, optimizer state, forward activation 
![[Pasted image 20250114232923.png]]

## 파라미터의 수의 장점
- 파라미터의 수가 많을수록 모델은 방대한 데이터셋을 활용해 학습할 수 있으며, 이를 통해 복잡한 언어 구조와 문맥적 관계, 언어 패턴들을 보다 정확하게 학습하고 재현할 수 있음 
- 단일 작업 뿐만 아니라 다양한 유형의 작업들을 수행하여 문서 생성, 요약, 질의 응답, 번역과 같은 다양한 작업에서 뛰어난 성능을 보여주고 있음 
- LLM 모델이 일반적인 지식에는 강한 모습을 보이지만, 특정 도메인에 대한 깊이 있는 분야에는 전문성을 가지기 어렵다는 한계점이 있음 
## 특정 작업에 특화된 모델 만들기 
- 기존의 사전 학습된 모델(Pre-trained Language Model)을 적절히 추가적으로 조정하여 목적에 맞게 조절 
- In-context Learning
	- 프롬프트 입력 시 관련 예시를 함께 제공 
	- 이 방식은 모델의 파라미터(가중치, 편향)는 업데이트 되지 않음 
- Full Fine-tuning
	- 모델의 모든 파라미터를 재학습하여 업데이트 
- PEFT
	- 선택된 일부 파라미터만 학습하여 업데이트 

# In-context Learning
- 사용자가 제공하는 몇 가지 예제만으로도 기본적인 추론을 수행
- 모델의 가중치는 고정되고, 프롬프트 엔지니어링만으로 문제 해결 
- 모델이 클래스 간의 관계나 속성을 통해 일반화하는 능력을 활용 
## Zero-shot Learning
- LLM에게 예시를 주지 않고 바로 질문을 하여 답을 얻는 방식 
- 모델이 학습 과정에서 본 적 없는 새로운 클래스를 인식할 수 있도록 하는 방법 
- 프롬프트에 작업에 대한 설명은 주어지지만, 예시는 제공되지 않음 
- 모델은 오직 작업 설명만을 바탕으로 추론을 수행함 
	- 새로운 설명 정보를 입력으로 제공 
	- 이미지의 특징을 설명하는 텍스트 정보를 사용하여 이전에 본 적 없던 이미지라도 해석이 가능
## One-shot Learning
- 각 클래스에 대해서 단 하나의 예시만 제공될 때 모델이 그 클래스를 인식할 수 있도록 학습하는 방법 
- 유사도 학습이나 메타 학습 등의 기법을 활용하여 구현  
	- 고양이를 인식하는 모델에게 새로운 종의 고양이를 인식하도록 요청한다면 
	- 모델은 이전에 본 적 없는 새로운 종의 고양이의 사진 하나만으로도 인식을 수행할 수 있어야 함 
	- 학습 데이터가 매우 제한적인 경우에 유용 
## Few-shot Learning
- 극소량의 데이터만을 이용하여 새로운 작업이나 클래스를 빠르게 학습하도록 설계된 알고리즘 
- 이 방법은 Meta Learning이나 학습 전략의 최적화등을 통해 적은 데이터로도 효과적인 일반화 능력을 갖추도록 함 



# LoRA
- LoRA는 큰 모델을 학습시켜야 하는 LLM 시대에 가장 사랑받는 PEFT 방법 중 하나임 
- LoRA는 모델 파라미터를 재구성(reparameterization)해 더 적은 파라미터를 학습함으로써 GPU 메모리 사용량을 줄일 수 있음 
- LoRA에서 파라미터 재구성은 행렬을 더 작은 2개의 행렬의 곱으로 표현해 전체 파라미터를 수정하는 것이 아니라 더 작은 2개의 행렬을 수정하는 것 
## 아이디어
- **모델 가중치의 변화량을 저차원(low-rank) 행렬로 표현**하여 fine-tuning 과정을 효율적 함 
- **Pre-trained 모델 가중치 고정:** 기존 LLM의 가중치(W)는 **고정(freeze)** 시켜 업데이트 하지 않음 
- **Low-Rank 행렬 추가:** 대신, 훨씬 작은 크기의 두 행렬 A와 B를 도입하여, 이 두 행렬의 곱 (ΔW = BA)으로 가중치 변화량을 표현
- **업데이트는 A와 B만:** Fine-tuning 과정에서는 **A와 B만 업데이트**합니다.

## 작동 방식
1. **Pre-trained 모델 (W):** 기존 LLM의 가중치는 그대로 유지됩니다.
2. **Low-Rank 행렬 (A, B):** 입력 x에 대해, `h = Wx + BAx` 와 같이 계산됩니다. 즉, 입력은 기존 가중치 W와 추가적인 변화량 BA를 모두 통과합니다.
3. **Fine-tuning:** A와 B는 무작위로 초기화된 작은 행렬이며, fine-tuning 과정에서 task-specific 데이터에 맞춰 학습됩니다.

![[Pasted image 20250115004923.png]]
- 추가한 파라미터를 기존 파라미터에 얼마나 많이 반영할지 결정하는 alpha가 존재 
- 위의 그림에서 행렬 A와 행렬 B부분을 (alpha/r)만큼의 비중으로 기존 파라미터 W에 더함 
- alpha가 16이고, r이 8이라면 행렬 A와 B에 (16/8) -> 2를 곱해 기존 기존 파라미터에 더함 
- alpha가 커질수록 새롭게 학습한 파라미터의 중요성을 크게 고려한다고 볼 수 있음 
- 학습 데이터에 따라 적절한 알파 값도 달라지기 때문에 실험을 통해 r값과 함께 적절히 설정 

### 셀프 어텐션의 예 
- 셀프 어텐션 연산의 query, key, value 가중치와 피드 포워드 층의 가중치와 같이 선형 연산의 가중치를 재구성 
- 이 중에서 특정 가중치에만 LoRA를 적용할 수도 있고, 전체 선형 층에 LoRA를 적용할 수도 있음 
- 보통 전체 선형 층에 LoRA를 적용한 경우 성능이 가장 좋다고 알려져 있으나, 이 부분 또한 실험을 통해 적절히 선택해야 함 
## 장점
- **효율적인 Fine-tuning:**
    - 업데이트해야 할 파라미터 수가 크게 줄어들어 **학습 속도가 빨라지고 메모리 사용량이 감소**
    - 훨씬 적은 계산량으로 fine-tuning이 가능하여 **비용 절감** 효과
- **과적합 방지:** 업데이트되는 파라미터 수가 적기 때문에 과적합 위험이 감소합니다.
- **원본 모델 성능 보존:** Pre-trained 모델 가중치를 고정함으로써, **원본 모델의 일반적인 지식과 능력을 보존**
- **빠른 Task 전환:** A와 B만 교체하면 되기 때문에, **다양한 task에 빠르게 적응**

## 단점:**
- **표현력 제한:** 가중치 변화량을 저차원 행렬로 제한하기 때문에, **표현력이 다소 제한**될 수 있습니다.
- **적절한 Rank 설정:** Low-rank 행렬의 차원(rank)을 적절하게 설정하는 것이 중요합니다. Rank가 너무 작으면 성능이 저하될 수 있고, 너무 크면 LoRA의 이점이 줄어듭니다.

## 활용
- **다양한 LLM Fine-tuning:** LoRA는 GPT, BERT, RoBERTa 등 다양한 LLM에 적용되어 fine-tuning 효율성을 높이는 데 사용됩니다.
- **PEFT (Parameter-Efficient Fine-Tuning):** LoRA는 어댑터(Adapters), Prefix-tuning 등과 함께 PEFT의 대표적인 기법 중 하나로, 효율적인 fine-tuning을 위한 다양한 연구에 활용됩니다.


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def load_model_and_tokenizer(model_id, peft=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if peft is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"":0})

    elif peft == 'lora':
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"":0})
        lora_config = LoraConfig(
                    r=8,
                    lora_alpha=32,
                    target_modules=["query_key_value"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print_gpu_utilization()
    return model, tokenizer
```

# QLoRA
- LoRA에 양자화를 추가해 메모리 효율성을 한 번 더 높은 학습 방법 
- 양자화란 기존 데이터를 더 적은 메모리를 사용하는 데이터 형식으로 변환하는 방법 
![[Pasted image 20250115010358.png]]
## 아이디어 
- 학습된 모델 파라미터는 거의 정규 분포에 가깝다고 알려져 있다. 
- LLama 모델을 확인 했을 때 92.5% 정도의 모델 파라미터가 정규 분포를 따랐다. 
- QLoRA 논문은 정규 분포를 활용하여 양자화하여 4비트 부동소수점 데이터 형식인 NF4를 제안 
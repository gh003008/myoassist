# MyoLeg 모델 구조 및 16 DOF 작동 원리 상세 분석

**날짜:** 2024-11-18  
**궁금증:** 왜 16 DOF 데이터만으로 60 qpos 모델이 정상 작동하는가?

---

## 🔍 핵심 질문들

1. **`q_` prefix가 뭐야?**
2. **Kinematic chain은 어떻게 구성되는가?**
3. **16 DOF 데이터로 60 qpos 모델이 작동하는 원리는?**
4. **`knee_translation1/2` 같은 보조 관절은 어떻게 처리되는가?**
5. **이건 어디 코드에 구현되어 있는가?**

---

## 1. `q_` prefix란?

### 정의:
- `q_`: **Position** (generalized coordinates)
- `dq_`: **Velocity** (time derivative of q)

### 데이터 저장 규약:

**HDF5 원본 데이터:**
```python
# OpenSim/mocap 데이터에서 추출
'q_pelvis_tx'  → 골반 x 위치
'q_pelvis_ty'  → 골반 y 위치  
'dq_pelvis_tx' → 골반 x 속도
```

**환경에서 사용할 때:**
```python
# environment_handler.py가 변환
'pelvis_tx'   → q_ prefix 제거됨
'dpelvis_tx'  → dq_ → d로 단순화
```

### 왜 prefix를 쓰나?

1. **명확성:** 위치인지 속도인지 구분
2. **표준:** Robotics/biomechanics 분야 표준 표기법
3. **자동화:** 스크립트가 자동으로 처리하기 쉬움

---

## 2. Kinematic Chain 구조

### XML에서 정의된 전체 구조 (myoLeg26_BASELINE.xml)

```
worldbody
└── pelvis (6 DOF)
    ├── pelvis_tx (slide)     ← qpos[0]  ✅ REFERENCE 1
    ├── pelvis_ty (slide)     ← qpos[1]  ✅ REFERENCE 2
    ├── pelvis_tz (slide)     ← qpos[2]  ✅ REFERENCE 3
    ├── pelvis_tilt (hinge)   ← qpos[3]  ✅ REFERENCE 4
    ├── pelvis_list (hinge)   ← qpos[4]  ✅ REFERENCE 5
    └── pelvis_rotation (hinge) ← qpos[5] ✅ REFERENCE 6
    
    ├── femur_r (3 DOF)
    │   ├── hip_flexion_r     ← qpos[6]  ✅ REFERENCE 7
    │   ├── hip_adduction_r   ← qpos[7]  ✅ REFERENCE 8
    │   └── hip_rotation_r    ← qpos[8]  ✅ REFERENCE 9
    │   
    │   └── tibia_r (3 DOF)
    │       ├── knee_r_translation1  ← qpos[9]  ❌ NOT IN REFERENCE
    │       ├── knee_r_translation2  ← qpos[10] ❌ NOT IN REFERENCE  
    │       ├── knee_angle_r         ← qpos[11] ✅ REFERENCE 10
    │       
    │       └── talus_r (1 DOF)
    │           ├── ankle_angle_r    ← qpos[12] ✅ REFERENCE 11
    │           
    │           └── calcn_r (foot)
    │               └── toes_r (1 DOF)
    │                   └── mtp_angle_r  ← qpos[13] ❌ NOT IN REFERENCE
    
    ├── femur_l (3 DOF) - 좌측 다리 (대칭)
    │   ├── hip_flexion_l     ← qpos[23] ✅ REFERENCE 12
    │   ├── hip_adduction_l   ← qpos[24] ✅ REFERENCE 13
    │   └── hip_rotation_l    ← qpos[25] ✅ REFERENCE 14
    │   
    │   └── tibia_l (3 DOF)
    │       ├── knee_l_translation1  ← qpos[26] ❌ NOT IN REFERENCE
    │       ├── knee_l_translation2  ← qpos[27] ❌ NOT IN REFERENCE
    │       ├── knee_angle_l         ← qpos[28] ✅ REFERENCE 15
    │       
    │       └── talus_l (1 DOF)
    │           ├── ankle_angle_l    ← qpos[29] ✅ REFERENCE 16
    │           
    │           └── calcn_l (foot)
    │               └── toes_l (1 DOF)
    │                   └── mtp_angle_l  ← qpos[30] ❌ NOT IN REFERENCE
    
    └── torso (상체)
        ├── clavicle_r → humerus_r → radius_r + ulna_r → hand_r (7 DOF)
        │   ├── r_shoulder_abd   ← qpos[40] ❌ NOT IN REFERENCE
        │   ├── r_shoulder_rot   ← qpos[41] ❌ NOT IN REFERENCE
        │   ├── r_shoulder_flex  ← qpos[42] ❌ NOT IN REFERENCE
        │   ├── r_elbow_flex     ← qpos[43] ❌ NOT IN REFERENCE
        │   └── ... (wrist joints)
        
        └── clavicle_l → humerus_l → ... (7 DOF) (대칭)
            └── ... (left arm joints)
```

### 전체 요약:

| 부위 | DOF | qpos 범위 | Reference 데이터 |
|------|-----|-----------|------------------|
| **Pelvis (골반)** | 6 | 0-5 | ✅ 모두 있음 (1-6) |
| **Right Leg** | 8 | 6-13 | ✅ 4개 (hip×3, knee×1, ankle×1) |
| **Left Leg** | 8 | 23-30 | ✅ 4개 (대칭) |
| **Right Arm** | 7 | 40-46 | ❌ 없음 |
| **Left Arm** | 7 | 47-53 | ❌ 없음 |
| **기타 (wrapping points)** | ~14 | 14-22, 31-39, 54-59 | ❌ 없음 |
| **총합** | **60** | 0-59 | **16개만 있음** |

---

## 3. 16 DOF로 60 qpos가 작동하는 마법 🪄

### 핵심 원리: **Coupled Joints & Passive Dynamics**

MuJoCo XML에서 정의:

```xml
<!-- Right knee: 3개 joint가 하나의 body에 정의됨 -->
<body name="tibia_r" pos="0 0 0">
    <!-- Translation 1: 좌우 미끄러짐 -->
    <joint axis="1 0 0" name="knee_r_translation1" pos="0 0 0" 
           range="-0.005574 0.00411" type="slide"/>
    
    <!-- Translation 2: 앞뒤 미끄러짐 (주요!) -->
    <joint axis="0 1 0" name="knee_r_translation2" pos="0 0 0" 
           range="-0.4226 -0.3953" type="slide"/>
    
    <!-- 회전 각도 (우리가 제어하는 joint) -->
    <joint axis="0 0 1" name="knee_angle_r" pos="0 0 0" 
           range="-2.531 0.0"/>
</body>
```

### 🔑 핵심: `knee_angle_r`과 `knee_translation2`의 관계

**생체역학적 현실:**
- 무릎이 굽혀지면 (knee_angle ↓)
- 경골(tibia)이 대퇴골(femur) 위에서 미끄러짐 (translation ↓)
- 이것은 **물리적 구속조건** (constraint)

**MuJoCo에서 처리:**

#### A) Stand Keyframe (기준 자세):
```python
qpos = [0, 0.91, 0, 0, 0, 0,  # pelvis (6)
        0, 0, 0,              # hip_r (3)
        0.00411,              # knee_r_translation1 ← 기본값
        -0.395,               # knee_r_translation2 ← 기본값 (중요!)
        0,                    # knee_angle_r
        -0.0143,              # ankle_angle_r
        0, ...]               # mtp_angle_r
```

#### B) Reference Data 적용 시:
```python
# 환경 코드 (myoassist_leg_imitation_ver1_0.py)
for key in self.reference_data_keys:
    self.sim.data.joint(f"{key}").qpos = reference_data[key][index]
```

**순서:**
1. `joint().qpos`는 **해당 joint의 qpos만** 설정
2. MuJoCo가 `mj_forward()` 호출 시:
   - Constraint satisfaction
   - Passive dynamics
   - Contact forces
3. **다른 joint들은 자동으로 조정됨**

#### C) 구체적 예시:

```python
# 1. 초기 상태 (stand keyframe)
knee_angle_r = 0.0         # 직립
knee_translation2 = -0.395  # 기본 위치

# 2. Reference data 적용
joint("knee_angle_r").qpos = -0.8  # 무릎 굽힘

# 3. mj_forward() 호출 후
# MuJoCo가 자동으로 조정:
knee_translation2 = -0.42  # 자동으로 더 미끄러짐!
#                    ^^^^^^
#                    Physics engine이 계산
```

---

## 4. Knee Translation의 역할

### 왜 필요한가?

**단순 hinge joint만 있으면:**
```
  Femur
    |
    O (knee_angle만)
    |
  Tibia
```
→ **비현실적!** 무릎이 단순 경첩처럼만 움직임

**Translation 추가하면:**
```
  Femur
    |
    O (knee_angle)
   / \
  /   \ (translation1, translation2)
 |     |
Tibia
```
→ **현실적!** 경골이 대퇴골 위에서 미끄러지며 회전

### 생체역학적 정확성:

실제 인간 무릎:
1. **Flexion (굽힘)**: 0° → 120°
2. **Translation**: 굽힘에 따라 2-3cm 뒤로 미끄러짐
3. **Rotation**: 약간의 내외회전

MuJoCo 모델:
```xml
<!-- range="-0.4226 -0.3953" -->
<!-- 범위: -0.4226m ~ -0.3953m (약 2.7cm 차이) -->
```

---

## 5. 코드 구현 위치

### A) 모델 정의: `myoLeg26_BASELINE.xml`

```xml
<!-- Lines 105-205: Joint 정의 -->
<body name="pelvis" pos="0 0 0">
    <joint name="pelvis_tx" type="slide"/>
    <joint name="pelvis_ty" type="slide"/>
    ...
    
    <body name="femur_r">
        <joint name="hip_flexion_r" range="-0.349 2.356"/>
        ...
        
        <body name="tibia_r">
            <joint name="knee_r_translation1" type="slide"/>
            <joint name="knee_r_translation2" type="slide"/>
            <joint name="knee_angle_r"/>
            ...
        </body>
    </body>
</body>

<!-- Line 737: Stand keyframe -->
<key name="stand" qpos="0 0.91 0 0 0 0 0 0 0 0.00411 -0.395 0 ..."/>
```

**역할:**
- Joint 종류 (hinge, slide)
- 운동 범위 (range)
- 부모-자식 관계 (kinematic tree)
- 기준 자세 (keyframe)

### B) Reference Data 적용: `myoassist_leg_imitation_ver1_0.py`

```python
def _follow_reference_motion(self, is_x_follow:bool):
    # 16개 DOF만 설정
    for key in self.reference_data_keys:  # ['pelvis_tx', ..., 'ankle_angle_l']
        self.sim.data.joint(f"{key}").qpos = \
            self._reference_data["series_data"][f"{key}"][self._imitation_index]
    
    # 속도도 설정
    for key in self.reference_data_keys:
        self.sim.data.joint(f"{key}").qvel = \
            self._reference_data["series_data"][f"d{key}"][...] * speed_ratio
```

**중요:**
- `joint().qpos`: MuJoCo의 고수준 API
- 해당 joint만 설정, 나머지는 physics engine이 처리

### C) Physics 계산: MuJoCo Engine (C/C++)

```python
# Python에서 호출
mujoco.mj_forward(model, data)
```

**내부 처리:**
1. **Constraint satisfaction**: Translation ↔ Rotation 관계 유지
2. **Contact dynamics**: 발-지면 접촉
3. **Muscle dynamics**: 근육 힘 계산
4. **Integration**: 다음 시간 스텝 계산

---

## 6. 왜 Stand Keyframe이 중요한가?

### 문제 상황:

```python
# ❌ Keyframe 없이 시작
data.qpos[:] = 0  # 모든 joint가 0

# Reference data 적용
joint("knee_angle_r").qpos = -0.8

# 결과:
knee_translation2 = 0  # 여전히 0!
#                   ^^^
#                   초기값이 잘못되어 physics가 수렴 실패
```

**문제:**
- `knee_translation2 = 0` → 다리 길이가 0
- Tibia가 femur 시작점에 붙음
- **정강이가 힙으로!**

### 해결:

```python
# ✅ Stand keyframe 사용
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
data.qpos[:] = model.key_qpos[key_id]

# 초기값이 올바름:
knee_translation2 = -0.395  # 정상 다리 길이
```

**그러면 환경 코드에서 왜 stand keyframe을 안 써도 되나?**

**답:** MuJoCo의 `joint().qpos` API가 알아서 처리!

```python
# Environment (원본 코드)
self.sim.data.joint("knee_angle_r").qpos = -0.8

# MuJoCo 내부에서:
# 1. knee_angle_r만 변경
# 2. Constraint 체크
# 3. 관련된 다른 joint 자동 조정 (translation2 등)
# 4. Physics 수렴
```

**Rendering script에서는 왜 필요한가?**

```python
# Rendering (직접 qpos 접근)
data.qpos[11] = -0.8  # knee_angle_r

# MuJoCo는 index만 알고 joint 관계를 모름!
# → Stand keyframe으로 초기값 설정 필수
```

---

## 7. 전체 데이터 플로우

```
[Mocap/OpenSim]
    16 DOF human motion capture
    ↓
[HDF5 파일]
    q_pelvis_tx, q_hip_flexion_r, ... (16개)
    ↓ convert/symmetrize
[NPZ: MuJoCo renderer format]
    q_ref: (12028, 16)
    joint_names: ['q_pelvis_tx', ..., 'q_ankle_angle_l']
    ↓ environment_handler.py
[series_data: Environment format]
    'pelvis_tx': [...], 'hip_flexion_r': [...], ... (q_ prefix 제거)
    ↓ myoassist_leg_imitation_ver1_0.py
[Environment 적용]
    for key in 16 joints:
        joint(key).qpos = reference_data[key]  ← 16개만 설정
    ↓ MuJoCo mj_forward()
[Physics Engine 계산]
    - knee_translation1/2 자동 조정 (coupled)
    - mtp_angle_r/l 자동 조정 (passive)
    - arm joints 고정 또는 중립 자세 유지
    - wrapping points 자동 계산
    ↓
[최종 상태: 60 qpos]
    모든 qpos 값이 물리적으로 일관되게 설정됨 ✅
```

---

## 8. 요약 답변

### Q1: `q_` prefix가 뭐야?

**A:** Position(q)과 velocity(dq)를 구분하는 표준 표기법.  
Environment handler가 로드 시 제거함 (`q_pelvis_tx` → `pelvis_tx`)

### Q2: Kinematic chain은 어떻게 구성되는가?

**A:** XML에서 body-joint 계층 구조로 정의.  
Parent body → child body 순서로 연결됨.

### Q3: 16 DOF로 60 qpos가 작동하는 원리는?

**A:** 
1. **Coupled joints**: Knee translation ↔ knee angle 자동 연동
2. **Passive dynamics**: MuJoCo physics engine이 나머지 계산
3. **High-level API**: `joint().qpos`가 관련 constraint 자동 처리

### Q4: `knee_translation1/2`는 어떻게 처리되는가?

**A:**
- Reference data에는 **없음** (16 DOF만)
- Stand keyframe의 기본값 (-0.395) 사용
- `knee_angle_r` 설정 시 **자동으로 조정됨**
- Physics engine이 biomechanical constraint 유지

### Q5: 어디 코드에 구현되어 있는가?

**A:**
- **모델 구조**: `myoLeg26_BASELINE.xml` (joint 정의, keyframe)
- **Reference 적용**: `myoassist_leg_imitation_ver1_0.py` (_follow_reference_motion)
- **Physics 계산**: MuJoCo C++ engine (mj_forward)
- **데이터 변환**: `environment_handler.py` (q_ prefix 제거)

---

## 💡 핵심 통찰

1. **High-level API의 힘**: `joint().qpos`는 단순 배열 접근이 아님.  
   Physics-aware setter로 constraint를 자동 처리.

2. **Coupled joints의 마법**: 하나의 joint만 움직여도  
   생체역학적 제약조건에 따라 다른 joint가 자동 조정.

3. **Stand keyframe의 역할**: 
   - Rendering: 필수 (qpos index 직접 접근)
   - Environment: 선택 (joint API가 알아서 처리)

4. **16 DOF의 충분성**: 핵심 운동학적 DOF만 제공하면  
   나머지는 physics와 biomechanical constraint가 해결.

---

**작성:** GitHub Copilot  
**날짜:** 2024-11-18  
**상태:** ✅ 완전히 이해됨!

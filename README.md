# Detecting-Abnormal-Lenders-from-Lending-Club-loan-data
I used machine learning, oversampling, and undersampling.

# Data
All Lending Club loan data from kaggle

(link: https://www.kaggle.com/datasets/wordsforthewise/lending-club?select=accepted_2007_to_2018Q4.csv.gz)

# 필요한 모듈 및 데이터 가져오기 + 데이터 나눠서 읽기
read_csv로 한꺼번에 얻으려고 시도했지만 데이터가 지나치게 많아서 구글코랩이 버티지 못했습니다. 따라서 데이터를 분할해서 read_csv 작업을 거쳤습니다. 조건은 다음과 같습니다.

1. 날짜가 2015년에서 2016년까지인 데이터만 가져오기

2. application_type == Individual인 데이터만 가져오기

3. loan_status에서 조건에 맞게 0과 1 매길 수 있는 데이터만 가져오기

추가로 분할해서 데이터 얻을 때 마지막 행이 항상 Total amount funded in policy code 2: 521953170 이라는 행으로 변환되는 이상현상이 있어서 이에 대해서도 대체했습니다.

# 데이터 전처리
종속변수를 제외한 모형 제외항목에 속한 변수 제거했으며, 종속 변수 데이터 분포 확인 결과 불균형 데이터였습니다. 0인 변수가 1인 변수보다 압도적으로 많음을 확인했으며, 1인 변수를 소수 또는 이상치로 생각해보았습니다. 그리고 각 변수에 대해 null값이 얼마 있는지 확인해보았습니다. null 확인 결과 크게 결측치가 40%인 변수들과 그렇지 않은 변수들이 있었습니다. 그래서 다음과 같은 작업을 진행하겠습니다.

1. 결측치가 20% 넘는 변수들의 경우 제거합니다.

2. 결측치가 20% 넘지않은 변수들의 경우 imputation 또는 표본 삭제 진행합니다.

3. 결측치 작업 전에, 연속형 변수에서 최소와 최대가 같은 변수가 무엇인지 확인해보겠습니다.

4. 최소와 최대가 같다면 하나의 상수로만 이뤄지고 있다는 의미이니 발견되면 삭제합니다.

5. object형 변수 / 숫자형 변수 구분해서 null에 따른 타 변수와 상관관계를 확인해보겠습니다.

위를 통해 알 수 있는 정보는 다음과 같습니다.

1. url은 쓸모가 없다. (애초에 접속도 안 된다.)

2. 은행카드 비율 75% 이상인지 아닌지에 따라 한도가 영향을 받는다.

그리고 나머지 결측치 있는 변수들의 정보를 확인하고 각각에 맞는 조치를 취하겠습니다. 한편, inq_last_6mths, mths_since_recent_inq, num_tl_120dpd_2m은 정수형 변수입니다. 또한, 기간이 정해져 있으므로 결측치에 대해 0으로 대체할 수 있다고 판단하였습니다. 나머지 변수인 revol_util, bc_open_to_buy, bc_util, mo_sin_old_il_acct, mths_since_recent_bc, num_rev_accts 등은 실수형 변수이므로 knn imputation을 진행합니다. imputation을 진행하기 전에 object 타입 변수가 있는지 확인해봅니다. 이제 나머지 변수들을 Imputation을 진행할 것입니다. 각 변수들의 상관관계를 확인해보았을 때 머신러닝을 이용해서 진행하려고 합니다. 상관관계가 어느정도 있는 것을 확인했으니 sklearn의 Imputation을 이용해도 된다는 결론을 얻었습니다. 그래서 IterativeImputer을 이용했습니다.

이제 변수들의 분포 확인해보겠습니다. 확인 결과 BoxPlot, DBSCAN, Isolation Forest, One-Class SVM 등 기존 Outlier Detection으로 소수점(이상치)들을 찾기 어려웠습니다. 왜냐하면 모든 변수들이 소수점들이 다수점 안에 포함되어 있기 때문입니다. 따라서 기존 Outlier Detection 방법론들은 해당 문제를 해결하기 적합한 방법론들은 아니라고 생각했습니다. 또한, 0 초과를 1 / 0 이하를 0으로 만들고 그에 따른 상관관계도 체크해봤습니다.

# 초기 상태에서 모델 학습
지금까지 한 데이터를 바탕으로 모델 학습을 시도하고자 합니다. 다만, 데이터 불균형 때문에 Class Weight를 65:14로 지정했습니다

# OverSampling 후 모델 학습
OverSampling은 여러가지 기법이 있지만, ADASYN을 도입하기로 했습니다. 소수점 관측치에 대해서 주변에 다수점이 많을 수록 더 많이 oversampling 하는 방법이라고 생각해 채택하였습니다. Border-line SMOTE는 보더라인에 가까우면 동일하게 샘플링을 많이 하지만, ADASYN은 주변 다수점(즉, 거리)에 따라서 샘플링 수가 달라집니다. OverSampling은 Train Data 한해서만 진행했습니다. 또한, 모델간의 비교를 위해 X_val, y_val인 Validation Data로 그대로 측정했습니다.

# UnderSampling 후 모델 학습
UnderSampling도 마찬가지로 여러가지 기법이 있지만, TomekLink 도입하기로 했음. CondensedNearestNeighbour도 시행했으나 지나치게 오래 걸려서 적용하기 힘들었습니다. UnderSampling도 Train Data 한해서만 진행했습니다. 또한, 모델간의 비교를 위해 X_val, y_val인 Validation Data로 그대로 측정했습니다.

# 결과 및 결론
F1-Score은 전체적으로 비슷했지만 Precision, Recall 등 다양한 요소를 분석했을 때 UnderSampling을 하고 DecisionTree으로 모델링한 쪽이 더 예측력이 높았습니다. Feature Importance를 통해 확인 결과 dti, tot_hi_cred_lim 등 순으로 영향력이 있었습니다.

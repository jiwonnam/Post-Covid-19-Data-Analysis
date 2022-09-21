# 코로나 너머 산업별 전망 엿보기
## - 카드, 배달, 소비지수, 온라인쇼핑몰, 소비자동향지수 데이터를  이용하여

---

코로나19는 전 세계 사람들의 삶을 빠르게 바꾸고 있습니다. 코로나19와 같은 전염병은 다른 시기보다 변화를 쉽게 받아들이게 만듦으로서, 이전에는 견고하게 보였던 개념과 가치들을 무너지게 만들기도하고 역으로 새로운 가치들을 떠오르게 만드는 동력이 되기도 합니다. 이로 인해 유망해 보였던 산업들이 위기를 겪기도 하고, 반대로 새로운 산업들이 떠오르기도 합니다.

2020년 7월, 전세계에서는 1000만명 이상의 코로나19 확진자가 나왔습니다. 여전히 코로나19 확진자는 세계적으로 증가 추세에 있고 백신이 개발, 임상 단계에 있는 것으로 볼 때 코로나 이후의 시대, 포스트 코로나 시대가 언제 올지는 알기 어렵지만, 이 글을 통해 사람들의 소비패턴을 분석함으로서 포스트 코로나 시대를 엿보고자 합니다.


저희는 코로나 확진자 데이터와 소비 관련 데이터를 이용하여 코로나19 시대에 사람들의 소비생활 변화를 분석하고 이를 바탕으로 큰 산업별로 포스트 코로나 시대의 모습을 예측해 보았습니다.

---

## 목 차
1. 전국 코로나19 확진자 수 추이
2. 카드 데이터 분석
3. 배달 데이터 분석
4. 소비지수 데이터 분석
5. 온라인 쇼핑몰 데이터 분석
6. 소비자동향지수 데이터 분석
7. 결 론

---

## 사용한 데이터

1. PatientInfo.csv - 대한민국 코로나19 확진자 정보
2. card.csv - 업종별 결재금액 데이터(서울시)
3. delivery.csv - 배달 호출 정보 데이터
4. index.csv - 품목별 소비지수 데이터
5. **(공공데이터)** [온라인 쇼핑몰 데이터(국가통계포털)](http://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_1KE10051&vw_cd=&list_id=&seqNo=&lang_mode=ko&language=kor&obj_var_id=&itm_id=&conn_path=)
6. **(공공데이터)** [소비자동향조사(국가통계포털)](http://kosis.kr/statHtml/statHtml.do?orgId=301&tblId=DT_040Y002&conn_path=I2)


```python
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import math
from datetime import date, timedelta

%matplotlib inline
from matplotlib import font_manager, rc
# 한글 폰트 불러오기
font_path = '/Library/Fonts/Arial Unicode.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
```

## 1. 전국 코로나19 확진자 수 추이
카드 소비량의 추이를 살펴보기 앞서, 코로나 유행 상황을 살펴보기 위해 전국 일별 확진자수 추이를 먼저 살펴보겠습니다.


```python
patients = pd.read_csv('KT_data_20200703/COVID_19/TimeProvince.csv', index_col='date')
patients_all = patients.groupby(['date'])['confirmed'].sum().to_frame('전국')
patients_all['서울'] = patients[patients['province'] == '서울'].confirmed
daily_all_confirmed = [patients_all['전국'].values[0]]
daily_seoul_confirmed = [patients_all['서울'].values[0]]

for i in range(1, len(patients_all.index)):
    daily_all_confirmed.append(patients_all['전국'].values[i] - patients_all['전국'].values[i-1])
    daily_seoul_confirmed.append(patients_all['서울'].values[i] - patients_all['서울'].values[i-1])

patients_all['전국'] = [a - b for a, b in zip(daily_all_confirmed, daily_seoul_confirmed)]
patients_all['서울'] = daily_seoul_confirmed
xlabel = ["2020-02-01","2020-03-01","2020-04-01","2020-05-01","2020-06-01"]

def find_loc(df, dates):
    marks = []
    for date in dates:
        marks.append(df.index.get_loc(date))
    return marks

sns.set(style="darkgrid", font=font_name)
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
ax = patients_all.sort_index(axis=1).plot(kind='bar', stacked=True, width=1, figsize=(16,5))
ax.xaxis.set_ticks(find_loc(patients_all, xlabel))
ax.xaxis.set_ticklabels(xlabel, rotation=0)
ax.set_xlabel('날짜')
ax.set_ylabel('확진자수(명)')
plt.title('전국 일자별 확진자수', fontsize=20);
```


    
![png](README_files/README_5_0.png)
    


그래프를 보면 2월 말부터 꾸준히 확진자가 발생하고 3월에 최다 확진자가 발생하였습니다.  
4월 부터 감소하는 모습을 보이다가 5월 황금연휴 이후 다시 발생하며, 6월에도 확진자가 꾸준히 발생하는 모습을 볼 수 있습니다.  
5월 이후부터는 서울의 코로나19 확진자의 수가 전국 확진자 수의 일정한 비율을 차지하며 꾸준히 확진자가 나타나고 있음을 알 수있습니다.

**이러한 상황 속에서 사람들의 소비는 어떻게 변화하고 있는지 본격적으로 카드 데이터를 살펴보겠습니다.**

## 2. 카드 데이터 분석
서울시 카드 데이터를 이용하여 코로나 시기에 비슷한 소비 패턴을 보이는 업종들을 찾아내고, 포스트 코로나 시기에 어떠한 양상을 보일지 분석해보았습니다.  
각 업종별로 20.01.04부터 20.06.14까지의 매출발생금액을 최고값으로 나누어 표준화하고 계층적 클러스터링을 하였습니다.  
(금액이 음수이거나 Nan인 경우는 0으로 처리하였습니다)


```python
card = pd.read_csv('KT_data_20200703/card.csv')
card = card[['receipt_dttm', 'adstrd_nm', 'mrhst_induty_cl_nm', 'selng_cascnt', 'salamt']]
card = card[card['selng_cascnt'].apply(lambda x: x.isnumeric())]
card = card[card['salamt'].apply(lambda x: x.isnumeric())]
card['selng_cascnt'] = card['selng_cascnt'].astype(int)
card['salamt'] = card['salamt'].astype(int)
card['receipt_dttm'] = card['receipt_dttm'].astype(str)

# 일별, 업종별 카드사용금액
day = card.groupby(['receipt_dttm','mrhst_induty_cl_nm'])['salamt'].sum().reset_index()
daily_sale_amount = day.pivot(index='mrhst_induty_cl_nm', columns='receipt_dttm', values='salamt')

for index, row in daily_sale_amount.iterrows():
    max_salamt = 0
    for col in daily_sale_amount.columns:
        if daily_sale_amount.loc[index,col] < 0 or math.isnan(daily_sale_amount.loc[index, col]):
            daily_sale_amount.loc[index, col] = 0
        if row.loc[col] > max_salamt:
            max_salamt = float(daily_sale_amount.loc[index, col])

    # Normalized by the max sale amount
    for col in daily_sale_amount.columns:
        if max_salamt != 0:
            daily_sale_amount.loc[index, col] = float(daily_sale_amount.loc[index, col]) / float(max_salamt)
        if max_salamt == 0:  # All salamts are negative or nan values
            daily_sale_amount.loc[index, col] = 0
```

### 2.1 서울시 카드 데이터 클러스터 맵


```python
clustergrid = sns.clustermap(daily_sale_amount, col_cluster=False, cmap='Blues', figsize=(12, 14));

# Clustering 결과 확인 후 Cluster를 네모 박스로 표시
sub_clusters = []
sub_clusters.append(('국산 신차', '홍삼 제품', 'r'))
sub_clusters.append(('초중고교육기관', '카지노', 'r'))
sub_clusters.append(('유흥주점', '사진관', 'r'))
sub_clusters.append(('한의원', '화물 운송', 'r'))
sub_clusters.append(('대형할인점', '한정식', 'r'))
sub_clusters.append(('중고자동차', '화방표구점', 'r'))

ax = clustergrid.ax_heatmap
xmax = len(daily_sale_amount.axes[1])
rows = list(daily_sale_amount.axes[0])
reordered_row = [rows[x] for x in clustergrid.dendrogram_row.reordered_ind]

for start, end, color in sub_clusters:
    # vertical lines
    ax.plot([0, 0], [reordered_row.index(start), reordered_row.index(end)+1], color, lw = 2.5)
    ax.plot([xmax-.1, xmax-.1], [reordered_row.index(start), reordered_row.index(end)+1], color, lw = 2.5)
    # horizontal lines
    ax.plot([0, xmax-.1], [reordered_row.index(start), reordered_row.index(start)], color, lw = 2.5)
    ax.plot([0, xmax-.1], [reordered_row.index(end)+1, reordered_row.index(end)+1], color, lw = 2.5)
ax.set_xlabel("날짜", fontsize=13);
ax.set_ylabel("업종", fontsize=13);
plt.ylabel('최고 소비금액 대비 비율');
```


    
![png](README_files/README_10_0.png)
    


저희는 위 클러스터링 결과를 6개의 클러스터 (빨간 박스)로 나누어 세부적으로 소비 패턴을 살펴보았습니다. 

분석에 앞서, 소비 패턴의 변화에는 다음과 같은 여러가지 원인이 있을 수 있습니다.
1. 기간 영향 (계절, 분기, 공휴일, 주말, 평일)
2. 사건 영향 (정책 (재난지원금), 시장, 외부요인)
3. <span style="color:red">코로나19 영향</span>

저희는 각 클러스터의 특징을 살펴보고 각 클러스터 내에서 <span style="color:red">코로나의 영향</span>으로 패턴 변화를 설명할 수 있는 업종들을 살펴보겠습니다.  


### **Cluster 1 - 관광, 호텔** 


```python
def plot_box(cluster_df, cluster_grid, box_points, show_cax=True, ylabel=""):
    sub_rows = list(cluster_df.axes[0])
    sub_reordered_row = [sub_rows[x] for x in cluster_grid.dendrogram_row.reordered_ind]
    if not show_cax:
        cluster_grid.cax.set_visible(False)
    cluster_grid.ax_col_dendrogram.set_visible(False)
    ax = cluster_grid.ax_heatmap
    xmax = len(daily_sale_amount.axes[1])
    for start, end, color in box_points:
        # vertical lines
        ax.plot([0, 0], [sub_reordered_row.index(start), sub_reordered_row.index(end)+1], color, lw = 3)
        ax.plot([xmax-0.4, xmax-0.4], [sub_reordered_row.index(start), sub_reordered_row.index(end)+1], color, lw = 3)
        # horizontal lines
        ax.plot([0, xmax-0.4], [sub_reordered_row.index(start), sub_reordered_row.index(start)], color, lw = 3)
        ax.plot([0, xmax-0.4], [sub_reordered_row.index(end)+1, sub_reordered_row.index(end)+1], color, lw = 3)
    ax.set_xlabel("날짜");
    ax.set_ylabel("업종");
    plt.ylabel(ylabel);
    
def get_sub_cluster(main_cluster, start, end, box_points=[]):
    rows = list(daily_sale_amount.axes[0])
    reordered_row = [rows[x] for x in main_cluster.dendrogram_row.reordered_ind]
    sub_cluster = pd.DataFrame(columns=daily_sale_amount.columns)
    for i in range(reordered_row.index(start), reordered_row.index(end)+1):
        sub_cluster = sub_cluster.append(daily_sale_amount.loc[reordered_row[i]])
    
    sub_clustergrid = sns.clustermap(sub_cluster, col_cluster=False, cmap='Blues', figsize=(14, len(sub_cluster.index)/3.5))
    plot_box(sub_cluster, sub_clustergrid, box_points, show_cax=False)

get_sub_cluster(clustergrid, start='국산 신차', end='홍삼 제품', box_points=[('관광여행', '항 공 사', 'r'), ('2급 호텔', '특급 호텔', 'b')])
```


    
![png](README_files/README_13_0.png)
    


### Cluster 1의 특징
**전체적으로는 특정 시기에 강하게 소비가 일어나고 이외에는 상대적으로 소비가 적게 일어나는 업종들로 보입니다.**  
**클러스터링 결과 비슷한 업종들이 (관광여행, 항공사), (2급, 1급, 특급호텔) 같이 묶인 것을 확인할 수 있습니다.**  

* <span style="color:red">관광여행과 항공사</span>는 1월말부터 소비가 줄어들어 6월까지 회복되지 않는 패턴을 보입니다.
* <span style="color:blue">호텔 업종 (2급 호텔, 1급 호텔, 특급 호텔)</span>은 2월부터 소비가 감소하다가 5월 부터 약간 회복됨을 보입니다.

---
### **Cluster 2 - 유치원, 면세점, 영화관**


```python
get_sub_cluster(clustergrid, start='초중고교육기관', end='카지노', box_points=[('외국인전용가맹점', '영화관', 'r')])

```


    
![png](README_files/README_16_0.png)
    


### Cluster 2의 특징
**Cluster1보다 산발적으로 소비가 특정 시기에 강하게 일어나는 업종들로 보입니다.**

* <span style="color:red">외국인전용가맹점, 유치원, 면세점, 영화관</span>은 서로 다른 업종임에도 불구하고 그룹화 된 것을 볼 수 있는데, 이는 코로나로 인한 여행 감소, 밀집된 장소를 기피하는 심리를 공통적으로 드러내는 것으로 보입니다.

---
### **Cluster 3 - 유흥주점, 음식점**


```python
box_points = [('유흥주점', '유흥주점', 'r'), ('단란주점', '단란주점', 'r'), ('노래방', '노래방', 'r')]
box_points.append(('당구장', '골프연습장', 'b'))
box_points.append(('중국음식', '일식회집', 'y'))
box_points.append(('미 용 원', '사진관', 'orange'))
get_sub_cluster(clustergrid, start='유흥주점', end='사진관', box_points=box_points)
```


    
![png](README_files/README_19_0.png)
    


### Cluster 3의 특징
**전반적으로 전체시기에 걸쳐 소비가 고르게 발생하는 모습을 보입니다.**

* <span style="color:red">유흥주점, 단란주점, 노래방</span>의 경우 특정 시기(하얗게 표시된 부분)에 소비가 거의 발생하지 않았습니다. 이것은 집단감염으로 인해 사람들이 해당 업종에 방문을 기피한 것으로 보입니다.
* <span style="color:blue">당구장, 스크린골프</span>의 경우 전체 시기동안 고른 소비를 보이며, <span style="color:blue">골프연습장</span>은 2월 이후에 점차 증가하는 추세를 보입니다.
* <span style="color:#EBD328">음식 관련 업종 (중국음식, 서양음식, 스넥, 주점, 일반한식, 일식횟집)</span>들은 비슷한 소비패턴을 보이고, 전반적으로 코로나가 유행하면서 소비가 감소했다가 4월 후반부터 서서히 회복하는 모습을 보이고 있습니다.
* <span style="color:orange">미용원, 사진관</span>은 주말에 소비가 주로 이루어지며, 소비가 주춤했다가 다시 회복하는 것으로 보아 코로나 유행 시기에 이용을 자제 하였다가 생활에 필요하기에 다시 이용하는 것이 아닐까 싶습니다.

---
### **Cluster 4 - 화장품, 주유소, 자동차관련**


```python
get_sub_cluster(clustergrid, start='한의원', end='화물 운송', box_points=[('화 장 품', '화 장 품', 'pink'), ('L P G', '주 유 소', 'b'), ('자동차시트/타이어', '카인테리어', 'r')])
```


    
![png](README_files/README_22_0.png)
    


### Cluster 4의 특징
**전반적으로 설날, 5월 연휴, 주말에 소비가 적게 일어나는 모습 (세로로 하얀 줄)을 보입니다.**

* <span style="color:pink">화장품</span>의 경우 소비가 감소하는 편이나, 특정 시기(3, 4, 5월 한번씩)에 소비가 몰리는 것으로 보아 기간세일등 외부적 요인에 영향이 큰 것으로 보입니다.
* <span style="color:blue">LPG, 주유소</span>는 거의 고르게 소비가 이루어지는 모습을 보입니다.
* <span style="color:red">자동차 시트/타이어, 카인테리어</span> 소비가 5월 이후 조금 늘어난 모습을 보입니다.

---
### **Cluster 5 - 의류용품**


```python
get_sub_cluster(clustergrid, start='대형할인점', end='한정식', box_points=[('기타의류', '안경', 'r'), ('일반 가구', '신   발', 'r'), ('골프경기장', '골프경기장', 'b')])
```


    
![png](README_files/README_25_0.png)
    


### Cluster 5의 특징
**전반적으로 4월 말부터 소비가 증가하는 모습을 보입니다.**

* <span style="color:red">기타의류, 안경, 일반가구, 내의판매점, 스포츠의류, 신발</span>의 경우, 감소세를 보이다가 5월에 소비가 증가하였습니다.  
코로나 유행 시기에 소비를 하지 않다가 이 시기에 갑자기 증가하는 것으로 보아 코로나 회복 영향과 더불어 재난지원금의 영향을 받은 것으로 생각됩니다.
* <span style="color:blue">골프경기장</span>은 고른 소비를 보인 cluster 3의 스크린골프, 골프연습장과는 다르게 소비가 점차 증가하는 모습을 보입니다.  
이는 계절적 영향과 함께, 코로나가 점차 통제됨에 따라 이용이 증가하는 것으로 추측됩니다. 

---
### **Cluster 6 - 학원**


```python
get_sub_cluster(clustergrid, start='중고자동차', end='화방표구점', box_points=[('보습학원', '헬스 크럽', 'r')])
```


    
![png](README_files/README_28_0.png)
    


### Cluster 6의 특징
**뚜렷한 특징은 없지만 소비가 꾸준히 발생하는 업종들로 보입니다.**

* <span style="color:red">보습학원, 예체능학원, 외국어학원</span> 등 학원업종이 같이 묶여있는 것을 확인할 수 있으며, 2월부터 감소하다가 4월 말부터 서서히 회복하는 추세를 보입니다.
* <span style="color:red">헬스 클럽, 회원제 형태의 업종</span>들도 학원 업종과 비슷한 양상을 보이고 있습니다.

### 2.2 소비패턴 클러스터 맵
각 클러스터에서 언급했던 업종들을 모아 소비패턴으로 그룹화하기 위해 Z-점수 정규화를 한 후 다시 클러스터링을 해보았습니다.


```python
def normalize_selected_items(selected_index):
    selected_items = pd.DataFrame(columns=daily_sale_amount.columns)
    for i in selected_index:
        selected_items = selected_items.append(daily_sale_amount.loc[i])

    for index, row in selected_items.iterrows():
        for col in selected_items.columns:
            if math.isnan(selected_items.loc[index, col]):
                selected_items.loc[index, col] = 0
        mean = np.mean(row)
        std = np.std(row)
        for col in selected_items.columns:
            selected_items.loc[index,col] = (selected_items.loc[index,col]-mean)/std
    return selected_items

selected_index = ['관광여행', '항 공 사', '2급 호텔', '1급 호텔', '특급 호텔', '유치원', '면 세 점', '영화관',
             '당구장', '스크린골프', '골프연습장', '중국음식', '서양음식', '스넥', '주점', '일반한식', '일식회집', '미 용 원', '사진관',
             '화 장 품', '자동차시트/타이어', '카인테리어', '기타의류', '안경', '일반 가구', '내의판매점', '스포츠의류', '신   발', '골프경기장',
             '보습학원', '예체능학원', '외국어학원', '레져업소(회원제형태)', '헬스 크럽']
selected_items = normalize_selected_items(selected_index)
selected_clustergrid = sns.clustermap(selected_items, col_cluster=False, cmap='RdBu', vmin=-5, vmax=5)
box_points = [('당구장', '사진관', 'k'), ('스크린골프', '내의판매점', 'k'), ('관광여행', '영화관', 'k'), ('주점', '헬스 크럽', 'k')]
plot_box(selected_items, selected_clustergrid, box_points, show_cax=True, ylabel="z-score")
```


    
![png](README_files/README_31_0.png)
    


위 클러스터맵 색상의 의미는 평균 소비금액을 0이라고 했을 때,  <span style="color:blue">파란색</span>은 평균보다 소비가 많았던 날들을 의미하고,  <span style="color:red">빨간색</span>은 평균보다 소비가 적었던 날들을 의미합니다.  
저희는 클러스터링 결과를 보고 해당 업종들을 4가지의 소비패턴으로 (검정색 박스) 분류해 보았습니다.


1. **큰 변화 없음**
2. **증가** (빨간색 -> 파란색)
3. **감소** (파란색 -> 빨간색)
4. **감소 후 증가** (파란색 -> 빨간색 -> 파란색)

**다음은 각 클러스터로 묶인 업종들의 소비 패턴을 꺾은선 그래프로 자세히 확인해보겠습니다.**

### 2.3 소비패턴 그래프


```python
def plot_selected_items(selected_index, title):
    selected_items = normalize_selected_items(selected_index)
    ax = selected_items.T.plot(figsize=(15, 5))
    plt.title(title, loc='left', fontsize=20)
    plt.xlabel("날짜", fontsize=14)
    plt.ylabel("z-score", fontsize=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=4)

# 1. 큰 변화 없음
selected_index = ['당구장', '미 용 원', '사진관']
plot_selected_items(selected_index, "큰 변화 없음")
# 2. 증가
selected_index = ['스크린골프', '골프연습장', '자동차시트/타이어', '카인테리어', '기타의류', '안경', '일반 가구', '내의판매점', '스포츠의류', '신   발', '골프경기장']
plot_selected_items(selected_index, "증가")
# 3. 감소
selected_index = ['관광여행', '항 공 사', '2급 호텔', '1급 호텔', '특급 호텔', '유치원', '면 세 점', '영화관']
plot_selected_items(selected_index, "감소")
# 4. 감소 후 증가
selected_index = ['중국음식', '서양음식', '스넥', '주점', '일반한식', '일식회집', '보습학원', '예체능학원', '외국어학원', '레져업소(회원제형태)', '헬스 크럽', '화 장 품']
plot_selected_items(selected_index, "감소 후 증가")
```


    
![png](README_files/README_34_0.png)
    



    
![png](README_files/README_34_1.png)
    



    
![png](README_files/README_34_2.png)
    



    
![png](README_files/README_34_3.png)
    


분류 결과, 각 그룹에서 다음과 같은 산업군들이 속해 있음을 확인할 수 있었습니다.

|**소비패턴**|**업종**|
|:-----:|:-----:|
|**증가**|스포츠관련, 인테리어, 자동차 용품, 의류|
|**감소**|여행 관련, 영화관, 유치원|
|**감소 후 증가**|음식점, 학원, 화장품, 헬스클럽|

다음으로는 카드 데이터에는 분류되지 않은 배달 산업의 추이를 배달 데이터를 이용하여 살펴보겠습니다.

## 3. 배달 데이터


```python
delivery = pd.read_csv('KT_data_20200703/delivery.csv', usecols=['PROCESS_DT','DLVR_REQUST_STTUS_VALUE','GOODS_AMOUNT'])
delivery = delivery[delivery['DLVR_REQUST_STTUS_VALUE']==1]  # 배달완료건
delivery['DLVR_COUNT'] = 1  # 배달건수
delivery = delivery.groupby(['PROCESS_DT']).agg({'DLVR_COUNT':'sum','GOODS_AMOUNT':'sum'}).reset_index()
delivery["PROCESS_DT"] = mdates.date2num(pd.to_datetime(delivery["PROCESS_DT"]))
```


```python
sns.set(style="whitegrid", font=font_name)
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15,10))
fig.suptitle("전국 일별 배달건수 및 배달금액", y=0.93, fontsize=20)
ax1 = sns.regplot(x="PROCESS_DT", y="DLVR_COUNT", data=delivery, marker='o', color='red', ax=ax1)
ax2 = sns.regplot(x="PROCESS_DT", y="GOODS_AMOUNT", data=delivery,  marker='x', color='blue', ax=ax2)

loc = mdates.AutoDateLocator()
ax1.xaxis.set_major_locator(loc)
ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
ax1.xaxis.label.set_visible(False); ax1.set_ylabel('배달건수', fontsize=14);
ax2.xaxis.set_major_locator(loc)
ax2.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
ax2.set_xlabel('날짜', fontsize=14); ax2.set_ylabel('배달금액', fontsize=14);
```


    
![png](README_files/README_38_0.png)
    


배달 데이터는 전체 지역을 포함하고 있으며, 배달건수와 배달금액 모두 전반적으로 코로나 시기에 **상승세**를 보이고 있습니다.  
**코로나 상황 속에서 확진자 증가 등의 요인으로 사람들이 배달 서비스를 경험하게 되고 계속 이용하는 것이 아닐까 생각하였습니다.**  
이에 배달 문화는 앞으로 상승하거나 지속될 가능성이 높다고 판단됩니다.

마지막으로 산업별 소비지수 변화를 살펴보겠습니다.

## 4. 서울시 산업별 소비지수 데이터

2019년 1월부터 2020년 5월까지 네 가지 산업의 **2018년 월평균 대비 매출 성장 비율**을 확인해보겠습니다.  
카테고리성장 지수는 100 이상이면 매출 상승, 이하면 하락을 의미합니다.


```python
index_df = pd.read_csv("KT_data_20200703/index.csv")
index_df['period'] = index_df['period'].astype(str)
index_df['age'] = index_df['age'].astype(str)
```


```python
# cgi: 카테고리성장지수 (2018년 월평균 대비 매출 성장 비율, 100을 기준으로 이상이면 매출 상승, 이하면 하락)
all_gender_grouped = index_df.loc[index_df['gender'] == 'all', :].groupby(['period', 'catl']).cgi.mean().reset_index()
all_gender_health = all_gender_grouped.loc[all_gender_grouped['catl'] == '건강/의료용품', :]
all_gender_food = all_gender_grouped.loc[all_gender_grouped['catl'] == '식품', :]
all_gender_daily = all_gender_grouped.loc[all_gender_grouped['catl'] == '일용품', :]
all_gender_cosmetic = all_gender_grouped.loc[all_gender_grouped['catl'] == '화장품', :]

plt.figure(figsize=(16,5));
sns.lineplot(x='period', y='cgi', data=all_gender_health, lw=3, label='건강/의료용품')
sns.lineplot(x='period', y='cgi', data=all_gender_food, lw=3, label='식품')
sns.lineplot(x='period', y='cgi', data=all_gender_daily, lw=3, label='일용품')
sns.lineplot(x='period', y='cgi', data=all_gender_cosmetic, lw=3, label='화장품')
plt.hlines(100, xmin='201901', xmax='202005', colors='k', linestyles='--')
plt.axvspan(13, 16, facecolor='r', alpha=0.2)
plt.title("전체 연령 산업 대분류별 평균 카테고리성장지수 (2018 월평균 대비)", size=20)
plt.xlabel("기준월", size=15)
plt.ylabel("카테고리성장지수", size=15)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=14)
plt.legend(fontsize=14);
```


    
![png](README_files/README_43_0.png)
    


전체적으로 모든 산업군이 2019년과 비교하여 코로나가 유행하면서 소비지수가 감소하는 것을 확인 할 수 있습니다.  
**건강/의료용품과 식품 산업의 경우는 다른 산업에 비해 감소폭이 적었으며, 화장품 산업의 경우 감소폭이 가장 컸습니다.**  
이 중 카드 데이터와 관련있는 산업군을 확인해보겠습니다.

### 4.1 건강/의료용품 산업 카테고리성장지수

건강/의료용품 산업의 카테고리성장지수를 연령별로 나누어 살펴보겠습니다.


```python
# Group by Age
twenties_grouped = index_df.loc[index_df['age']=='20', :].groupby(['period', 'catl']).cgi.mean().reset_index()
thirties_grouped = index_df.loc[index_df['age']=='30', :].groupby(['period', 'catl']).cgi.mean().reset_index()
forties_grouped = index_df.loc[index_df['age']=='40', :].groupby(['period', 'catl']).cgi.mean().reset_index()
fifties_grouped = index_df.loc[index_df['age']=='50', :].groupby(['period', 'catl']).cgi.mean().reset_index()
sixties_grouped = index_df.loc[index_df['age']=='60', :].groupby(['period', 'catl']).cgi.mean().reset_index()
all_ages_grouped = index_df.loc[index_df['age']=='all', :].groupby(['period', 'catl']).cgi.mean().reset_index()

health_20 = twenties_grouped.loc[twenties_grouped['catl'] == '건강/의료용품', :]
health_30 = thirties_grouped.loc[thirties_grouped['catl'] == '건강/의료용품', :]
health_40 = forties_grouped.loc[forties_grouped['catl'] == '건강/의료용품', :]
health_50 = fifties_grouped.loc[fifties_grouped['catl'] == '건강/의료용품', :]
health_60 = sixties_grouped.loc[sixties_grouped['catl'] == '건강/의료용품', :]
health_all = all_ages_grouped.loc[all_ages_grouped['catl'] == '건강/의료용품', :]

plt.figure(figsize=(16, 5))
plt.plot(health_20['period'], health_20['cgi'], 'o-', label="20s")
plt.plot(health_30['period'], health_30['cgi'], 'o-', label="30s")
plt.plot(health_40['period'], health_40['cgi'], 'o-', label="40s")
plt.plot(health_50['period'], health_50['cgi'], 'o-', label="50s")
plt.plot(health_60['period'], health_60['cgi'], 'o-', label="60s")
plt.plot(health_all['period'], health_all['cgi'], 'o-', label="all_ages")
plt.hlines(100, xmin='201901', xmax='202005', colors='k', linestyles='--')
plt.axvspan(13, 16, facecolor='r', alpha=0.2)
plt.title("건강/의료용품 소비지수", fontsize=20)
plt.xlabel("기준월", fontsize=15)
plt.ylabel("카테고리성장지수", fontsize=15)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=14)
plt.legend(fontsize=14);
```


    
![png](README_files/README_46_0.png)
    


코로나 유행 이후 20대를 제외하고 모든 연령에서 소비지수가 감소하고 있습니다. 반면 20대는 2019년부터 코로나 이후에도 매출 상승세를 보이고 있습니다.  
**이러한 상승 추세의 원인은 20대가 부모님 세대를 대신하여 건강/의료용품을 구매한 경우가 늘거나, 실제로 건강/의료용품에 대한 관심이 많아졌기 때문일 수 있습니다.**

### 4.2 화장품 산업 카테고리 성장지수

화장품 산업의 카테고리성장지수를 연령별로 나누어 살펴보겠습니다.


```python
beauty_20 = twenties_grouped.loc[twenties_grouped['catl'] == '화장품', :]
beauty_30 = thirties_grouped.loc[thirties_grouped['catl'] == '화장품', :]
beauty_40 = forties_grouped.loc[forties_grouped['catl'] == '화장품', :]
beauty_50 = fifties_grouped.loc[fifties_grouped['catl'] == '화장품', :]
beauty_60 = sixties_grouped.loc[sixties_grouped['catl'] == '화장품', :]
beauty_all = all_ages_grouped.loc[all_ages_grouped['catl'] == '화장품', :]
beauty_grouped_by_age = [beauty_20, beauty_30]

plt.figure(figsize=(16, 5))
plt.plot(beauty_20['period'], beauty_20['cgi'], 'o-', label="20s")
plt.plot(beauty_30['period'], beauty_30['cgi'], 'o-', label="30s")
plt.plot(beauty_40['period'], beauty_40['cgi'], 'o-', label="40s")
plt.plot(beauty_50['period'], beauty_50['cgi'], 'o-', label="50s")
plt.plot(beauty_60['period'], beauty_60['cgi'], 'o-', label="60s")
plt.plot(beauty_all['period'], beauty_all['cgi'], 'o-', label="all_ages")
plt.hlines(100, xmin='201901', xmax='202005', colors='k', linestyles='--')
plt.axvspan(13, 16, facecolor='r', alpha=0.2)
plt.title("나이대별 화장품 소비지수 변화", fontsize=20)
plt.xlabel("기준월", fontsize=15)
plt.ylabel("카테고리성장지수", fontsize=15)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=14)
plt.legend(fontsize=14);
```


    
![png](README_files/README_49_0.png)
    


화장품 산업은 코로나 유행 이후 모든 연령대에서 소비지수가 감소하는 것을 확인 할 수 있습니다. **눈에 띄는 것은 20, 30대의 소비지수 변화인데, 2019년에 비해 코로나 유행 후 가장 큰 감소 폭을 보인다는 점에서 코로나의 영향을 많이 받은 나이대라고 볼 수 있습니다.**

## 5. 온라인 쇼핑몰 데이터

**카드 데이터는 2020년 1월부터의 데이터를 담고 있어, 카드 데이터만으로는 소비 패턴의 변화가 시기적 원인인지 코로나의 원인인지 확인하기 어려웠습니다.**  
이러한 이유로 저희는 코로나 이전 시기를 포함하는 산업군 별 데이터를 찾아보았고, **국가통계포털에서 2018년부터의 온라인 쇼핑몰 데이터를 사용하였습니다.**  
온라인쇼핑몰 데이터가 전체 소비의 일부이고, 물가지수를 고려하진 않았지만 소비패턴을 확인하는데에는 무리가 없다고 생각하였습니다.  

### 5.1 온라인쇼핑몰 주요품목 합계 거래액

먼저 온라인쇼핑몰 주요품목 합계 거래액을 살펴보겠습니다.  
저희는 주요품목으로 위에서 살펴본 산업군들에 해당하는 의류, 스포츠·레저용품, 화장품, 자동차 및 자동차용품, 가구, 여행 및 교통서비스, 음식서비스를 선정하였습니다.


```python
online_shopping = pd.read_csv('online_mall.csv', index_col='상품군별')
online_shopping[online_shopping.select_dtypes(include=['number']).columns] /= 100  # 단위: 백만원->억
online_shopping = online_shopping.T[['의복','신발','가방','패션용품 및 액세서리','스포츠·레저용품','화장품','자동차 및 자동차용품','가구','여행 및 교통서비스','음식서비스']]
online_shopping['의류'] = online_shopping.loc[:,'의복':'패션용품 및 액세서리'].sum(axis=1)  # 의복~패션용품을 의류로 통합
online_shopping = online_shopping.drop(columns=['의복','신발','가방','패션용품 및 액세서리'])
online_shopping['주요품목합계'] = online_shopping[list(online_shopping.columns)].sum(axis=1)

ax = online_shopping.loc[:, online_shopping.columns=='주요품목합계'].plot(figsize=(15,5), style='.-', ms=10, lw=3, legend=False)
ax.set_xticks(range(len(online_shopping.index)))
ax.set_xticklabels(online_shopping.index.tolist(), rotation=45)
ax.axvspan(0, 4, facecolor='gray', alpha=0.2)
ax.axvspan(12, 16, facecolor='gray', alpha=0.2)
ax.axvspan(24, 28, facecolor='r', alpha=0.2)

plt.title("온라인쇼핑몰 주요품목 합계 거래액", fontsize=20)
plt.xlabel("년월 (201801~202005)", fontsize=15)
plt.ylabel("거래액(단위:억)", fontsize=15)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=14);
```


    
![png](README_files/README_53_0.png)
    


온라인쇼핑몰 주요품목의 합계 거래액을 보면 2018년부터 2020년 5월까지 증가하는 추이를 확인할 수 있습니다.  
코로나19의 영향으로 2020년 1, 2월에 2019년 동월 대비 대폭 감소하긴 하였지만, 비슷한 패턴으로 3월 이후 다시 증가하고 있습니다.  
전체적인 증가 추세와, 코로나 유행 시기에도 다시 증가하고 있음을 감안할 때, 앞으로도 온라인 쇼핑몰 거래액은 증가할 것으로 예상됩니다.  

다음으로는 각 주요품목별 거래액을 살펴보겠습니다.

### 5.2 온라인 쇼핑몰 상품군별 거래액


```python
ax = online_shopping.loc[:, online_shopping.columns!='주요품목합계'].plot(figsize=(16,8), style='.-', ms=10, lw=3)
ax.axvspan(1, 4, facecolor='gray', alpha=0.2)
ax.axvspan(13, 16, facecolor='gray', alpha=0.2)
ax.axvspan(25, 28, facecolor='r', alpha=0.2)
ax.set_xticks(range(len(online_shopping.index)))
ax.set_xticklabels(online_shopping.index.tolist(), rotation=45)

plt.title("온라인쇼핑몰 상품군별 거래액", fontsize=20)
plt.xlabel("년월 (201801~202005)", fontsize=15)
plt.ylabel("거래액(단위:억)", fontsize=15)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=14)
plt.legend(fontsize=14);
```


    
![png](README_files/README_56_0.png)
    


코로나시기에 해당하는 월을(빨간색, 2월-5월) 2018, 2019년과 비교하기 위해 같은 시기를 회색으로 표시하였습니다. 

거래액이 큰 상품군부터 살펴보면
- **<span style="color:#db93c7">의류</span>**: 2018, 2019년과 비교하여 2020년 5월에 거래액이 급격히 상승한 것으로 보아, 재난지원금의 영향 또는 소비심리의 회복으로 해석할 수 있을 것 같습니다.  

- **<span style="color:#755aa1">여행 및 교통 서비스</span>**: 카드 데이터에서 보여지듯, 2020년 2월에 거래액이 급락하는 것을 볼 수 있습니다.  

- **<span style="color:#f0a535">화장품</span>**: 시기적 패턴으로 봤을때 3월에 상승했어야 하나, 계속 감소하는 것으로 보아 코로나의 영향을 받아 소비가 위축된 것으로 보입니다.  

- **<span style="color:#80581c">음식서비스</span>**: 음식서비스는 온라인 주문 후 조리되어 배달되는 음식(치킨, 피자 등 배달서비스)을 의미합니다. 배달데이터에서 살펴본 건수, 금액의 증가 추이와 비슷하게 코로나 시기에도 거래액이 증가하고 있습니다. 또한 2018년, 2019년에 비해 같은 시기에 증가폭이 큰 것을 확인할 수 있습니다.  

- **<span style="color:#3958d4">스포츠,레저용품</span>**: 2018, 2019년 동월과 비슷하게 거래액이 증가하고 있으나 증가폭이 더 높습니다.  

- **<span style="color:#db4a2a">가구</span>**: 스포츠, 레저용품과 비슷하게 전년과 비교하여 코로나 시기에 거래액 증가폭이 더 크고, 이후에도 거래액이 유지되는 것을 볼 수 있습니다.  

- **<span style="color:#089e26">자동차 및 자동차용품</span>**: 전년도에는 큰 변화가 없었던 것에 반해, 코로나가 한창 유행한 2, 3월에 거래액이 상승했습니다.

**코로나19로 비대면 서비스들이 활발해지고 있는것으로 볼 때, 각 산업에서 온라인 판매를 도입하고 활성화한다면 위축되었던 소비를 회복하는 데 도움이 될 것이라 생각합니다.**

##  6. 소비자동향지수

**소비자 동향지수(CSI)는 소비자의 심리를 반영한 심리지표**로서 0에서 200까지의 값을 가질 수 있는데 지수가 **100이상(이하)이면 긍정(부정)적인 답변을 한 소비자가 부정(긍정)적인 답변을 한 소비자보다 많다는 것을 의미**하며 아래와 같은 식으로 계산됩니다.

100 + 100×(매우긍정×1.0＋다소긍정×0.5－다소부정×0.5－매우부정×1.0)/전체 응답가구수
<!-- $$100 + 100 × \frac{매우긍정×1.0＋다소긍정×0.5－다소부정×0.5－매우부정×1.0}{전체 응답가구수}$$ -->

이는 소비자의 경제상황에 대한 판단과 향후 소비지출계획 등을 파악하여 경제현상 진단 및 전망에 활용되는 데이터로, 코로나19 시기 소비자 심리변화를 통해 포스트 코로나 시대의 산업별 소비자 심리를 예측하고자 했습니다.

### 6.1 소득구간별 소비자동향지수
**저희는 소득구간별로 소비자의 심리 및 지출 전망이 어떻게 변화하는지 살펴보았습니다.**


```python
consumer = pd.read_csv('consumer_csi.csv')
consumer = consumer.loc[consumer['지수코드별'].isin(['의류비 지출전망CSI', '외식비 지출전망CSI', '여행비 지출전망CSI', '교육비 지출전망CSI', '의료·보건비 지출전망CSI', '교양·오락·문화생활비 지출전망CSI'])]
consumer= consumer.loc[consumer['분류코드별'].isin(['100만원미만','100-200만원','200-300만원','300-400만원','400-500만원','500만원이상'])]

group_list = consumer['지수코드별'].unique().tolist()

fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
for i in range(len(group_list)):
    group = consumer.loc[consumer['지수코드별']==group_list[i]].loc[:,'분류코드별':'202006'].set_index('분류코드별').T
    ax = group.plot(ax=axes[int(i/2), int(i%2)], title=group_list[i], figsize=(18, 15), lw=3)
    ax.title.set_size(20)
    ax.axvspan(13, 17, facecolor='r', alpha=0.2)
    ax.set_xticks(range(len(group.index)))
    ax.set_xticklabels(group.index.tolist(), rotation=45)
    ax.set_ylabel('소비자동향지수(CSI)', fontsize=14)
    ax.legend(loc="lower left")
plt.suptitle("소비자동향지수", y=0.92, fontsize=25);
```


    
![png](README_files/README_60_0.png)
    


**위의 그래프는 소득구간을 100만원 단위로 나누어 6개의 구간(100미만, 100-200, 200-300, 300-400, 400-500, 500만원 이상)의 2019년부터 소비자동향지수의 변화를 보여줍니다.**   

**여행비** : 코로나 유행 이후 지수가 가장 많이 떨어졌으며 100이하인 상태로 오래도록 유지되고 있습니다. 이는 사람들의 여행에 대한 소비 심리가 아직 회복되지 않음을 보여줍니다. 코로나 이전 시기에는 소득구간별 소비자동향지수의 차이가 컸지만, 코로나 이후에는 그 차이가 줄어들어 소득구간에 큰 상관없이 여행비 지출전망은 낮은 것으로 나타났습니다.  

**교육비** : 코로나 시기에 지수가 하락하긴 하였지만, 다른 부분에 비해 감소폭이 가장 적으며 코로나 이전과 비슷한 양상을 유지하고 있습니다. 소득이 낮은 가구도 외식비 등 다른 항목에 비하면 교육비에 대한 지출은 아끼지 않는 것으로 보입니다. 특히, 300만원 이상인 가구는 코로나 시기에도 100이상으로 교육비에 대한 지출을 하겠다고 긍정적인 답변을 많이 내놓았습니다. **그러나 300만원 이상과 100만원 미만 가구의 격차가 다른 항목에 비해 크게 유지되는 것으로 보아, 이러한 상황이 지속될 경우 경제적 차이에 따른 교육 격차가 더욱 커질수 있다고 생각됩니다.**  

**의류비 및 외식비** : 여행비 만큼은 아니지만, 코로나 유행시기에 전반적으로 지수가 감소하였고, 4월 이후 점차 회복하고 있습니다. 이를 통해, 사람들이 의류, 외식에 대한 심리가 서서히 회복되고 있음을 알 수 있습니다.  

**의료, 보건비** : 소득 구간에 상관없이 의료비에 대한 지출을 긍정적으로 전망함을 보여줍니다. 코로나 시기에도 소비자동향지수가 110선을 유지하고 있어 사람들이 건강 관련 지출을 중요하게 생각하고 있음을 알 수 있습니다.  

**교양, 오락 및 문화생활비** : 전반적으로 코로나 시기에 감소하였으며, 다른 의류비나 외식비에 비해 회복이 더딤을 보여줍니다.

### 6.2 코로나 전후 소비자동향지수 차이

다음으로 각 항목별 **코로나 전(2019.01-2020.01) 후(2020.02-2020.06)**의 소비자동향지수의 차이를 살펴봄으로써, 코로나 전후로 각 항목별로 지출전망이 어느정도 변화하였는지 알아보고자 합니다.   


```python
consumer = consumer.groupby('지수코드별').mean()
consumer['코로나전'] = consumer.loc[:, '201901':'202001'].mean(axis=1)
consumer['코로나후'] = consumer.loc[:, '202002':'202006'].mean(axis=1)
consumer['차이'] = consumer['코로나전'] - consumer['코로나후']
ax = consumer.sort_values('차이', ascending=True).iloc[:, -1].plot.bar(figsize=(15,7))
plt.xlabel("", fontsize=15)
plt.ylabel("소비자동향지수 차이", fontsize=15)
plt.xticks(fontsize=14, rotation=45)
plt.title('항목별 코로나 전후 소비자동향지수 차이', fontsize=20);
plt.hlines(consumer['차이'].mean(), -10, 10, colors='k', linestyles='--', lw=3, label='평균')
plt.legend(fontsize=15);
```


    
![png](README_files/README_63_0.png)
    


**코로나 전과 후에 소비자동향지수의 차이가 적다는 것은 소비자들의 지출전망이 크게 달라지지 않음을 의미하며, 코로나로 인한 영향을 적게 받았음을 보여줍니다.**  
차이값들의 평균보다 크게 적은 두가지 항목은 의료,보건비와 교육비입니다. 이를 통해 코로나 시기에도 사교육(교육비)에 대한 소비 심리는 크게 위축되지 않음을 보여줍니다.   
여행비는 코로나 전후 사람들의 지출 계획이 가장 크게 줄어든 것으로 보아, 코로나로 여행 산업이 크게 타격받았음 다시 한번 보여줍니다.

![WHO](https://www.who.int/universal_health_coverage/choosing-necessities.gif "WHO")
출처: [World Health Organization](https://www.who.int/universal_health_coverage/choosing-necessities.gif)

## 6. 결론

지금까지 카드 데이터, 배달 데이터, 소비지수, 온라인쇼핑몰, 소비자동향지수 데이터를 살펴보았습니다. 이를 종합하여 포스트 코로나 시대의 각 산업에 대한 생각을 정리해보겠습니다.

* **음식 산업**  
대부분의 음식점들이 코로나 시기에 감소하였다가 4월 후반부터 회복한 것으로 보아 포스트 코로나 시대에도 이전과 비슷한 소비가 이루어질 것으로 예상됩니다.  
배달산업은 코로나 시기에 꾸준히 증가하는 것으로 보아 이후에도 증가하거나 비슷한 양상을 보일 것으로 예상됩니다.  


* **교육 산업**  
유치원은 불안감에서인지 소비가 아직 회복을 못한 반면 보습, 외국어, 예체능 학원의 경우 감소하였다가 회복하는 모습을 보였습니다. 이를 통해 코로나와 같은 전염병이 사교육 학원에 대한 열기를 식히지는 못하였으며 앞으로도 학원에 대한 소비는 유지 될 것이라 생각합니다.  


* **여행 산업**  
해외 여행 제한으로 인해 항공, 여행 산업이 감소하는 것은 어느정도 예상한 바이며, 실제 데이터에서도 감소 후 회복하지 못하는 모습을 보이고 있습니다. 포스트 코로나 시대에는 이것이 지속 될지, 오히려 보상심리로 여행이 증가할지 예측하기는 어렵습니다. 코로나 시기의 경영난을 이겨내면 다시 살아날 가능성이 있지만, 이러한 시기가 장기화될 경우 전체적인 산업군의 축소를 야기하여 여행 산업이 위축될 수 있다고 생각합니다.  


* **스포츠/건강 산업**  
코로나 유행시기에도 스크린골프, 골프연습장의 이용은 지속되는 것으로 보아 골프의 경우 포스트 코로나 시기에도 비슷한 양상을 보일 것으로 예상됩니다. 그러나 다른 스포츠에 대한 구체적인 데이터의 구분은 없기에 전반적인 스포츠 산업의 양상을 예측하기는 어렵습니다.
소비지수 데이터로 확인한 건강/의료 산업의 경우에는 2019년에 전년대비 상승세를 보였으며 코로나 시기에도 감소폭이 다른 산업에 비해 가장 적었으므로 포스트 코로나 시기에 충분히 회복 후 성장할 가능성이 있다고 생각합니다.  


* **인테리어 산업**  
일반 가구, 주방용 식기, 카인테리어, 자동차타이어/시트의 소비가 서서히 증가하는 것으로 보아 재택근무, 원격수업 등으로 집에서 생활하는 시간이 많아지고, 대중교통 대신 개인 차를 많이 이용하면서 실내 인테리어, 카인테리어에 대한 관심도 많아졌다고 봅니다. 포스트 코로나 시기에 재택근무, 원격수업, 원격진료 등 비대면, 개인의 생활이 전보다 활발해지면 개인의 공간에 대한 관심으로 이어져 인테리어 산업이 증가할 수 있다고 생각합니다.  


* **의류 산업**  
의류, 신발의 경우 코로나가 한참 유행했던 2, 3, 4월의 경우 소비가 적게 이루어지다가 5월부터 급격하게 증가하는 모습을 보였습니다. 이는 재난지원금의 영향 또는 위축되었던 소비심리의 회복으로 볼 수 있을 것입니다.


* **화장품 산업**  
카드 데이터와 소비지수 데이터를 종합적으로 본 결과 화장품의 경우 전반적으로 소비가 감소했으나 한달에 한번씩 주기적으로 큰 소비가 발생하는 것으로 보아 20, 30대가 세일 기간에 주로 소비를 하는 것으로 보였습니다. 화장품 산업은 20대, 30대를 타겟으로 하는 세일 등으로 포스트 코로나시대에 돌파구를 찾을 수 있지 않을까 생각합니다.     

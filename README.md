<< 파일구조 >>

main.py: 메인파일, 메인을 실행시키면 동작합니다.

bin   
 ㄴ config.conf: 데이터 폴더 경로, aws 설정, 모델 설정 등의 정보   
 ㄴ logger.yaml: 로그 정보과 관련한 파일
 
dataset   
 ㄴ training (data + labels)   
 ㄴ validation (data + labels)   
 
logs   
 ㄴ cocon.log: 로그

model   
 ㄴ MultiLabelBinarizer   
 ㄴ saved_model.pb : 학습하여 저장된 모델
   
plot   
 ㄴ 학습과 관련한 그래프 저장
 
util   
 ㄴ classify.py: 인퍼런스   
 ㄴ dataset.py: 데이터셋 저장   
 ㄴ train.py: 모델 트레인   
 ㄴ vggnet.py: 백본 모델 



 
 
<< 실행 방법 >>
1. bin/config.conf 의 데이터 폴더 경로, aws 설정, 모델 설정 등의 정보를 기입
2. main.py 실행
3. saved_model.pb 저장된 모델에 의해 validation 데이터의 데이터와 예측 값이 출력 

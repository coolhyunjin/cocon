import boto3


def load_dataset(accessKey, secretKey, bucket_name, dataset_path):

    # 1. s3 접근
    client = boto3.client('s3', aws_access_key_id=accessKey, aws_secret_access_key=secretKey)

    # 2. 버킷의 데이터 파악
    paginator = client.get_paginator("list_objects_v2")
    # 1000 개씩 반환되는 list_objects_v2의 결과 paging 처리를 위한 paginator 선언

    # 3. 버킷의 데이터 저장 (총 이미지 825,510)
    for page in paginator.paginate(Bucket=bucket_name):
        contents = page["Contents"]
        for i in range(len(contents)):
            file_path = contents[i]['Key']
            folder_nm_lst = file_path.split('/')
            folder_nm, type_nm = folder_nm_lst[0], folder_nm_lst[1]
            client.download_file(bucket_name, contents[i]['Key'],
                                 dataset_path + '/{}/{}/{}'.format(folder_nm, type_nm, folder_nm_lst[-1]))

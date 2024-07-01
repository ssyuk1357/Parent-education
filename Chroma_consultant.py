import pandas as pd
import chromadb
import ast

def get_empathy_context(query_text):
    # 하드코딩된 경로와 컬렉션 이름
    csv_path = 'empathy_dialogue.csv'
    db_path = 'your_database_directory1'
    collection_name = 'my_collection'

    # 1. CSV 파일을 pandas DataFrame으로 읽어오기
    df = pd.read_csv(csv_path)

    # 2. Chroma DB 인스턴스 생성
    client = chromadb.PersistentClient(path=db_path)  # provide a path to persist your database

    # 컬렉션 생성 또는 선택
    collection = client.get_collection(collection_name)

    # Query ChromaDB for similar documents based on the query text
    results = collection.query(
        query_texts=[query_text],
        n_results=5704  # 원하는 검색 결과의 수 (충분히 큰 수로 설정)
    )

    # 결과 정렬 (유사도 점수에 따라 높은 순서로)
    sorted_results = sorted(zip(results['documents'][0], results['distances'][0], results['ids'][0], results['metadatas'][0]),
                            key=lambda x: x[1], reverse=True)  # 거리 값으로 정렬 (높은 순서)

    # 'role'이 '자녀'인 문서 필터링
    filtered_results = []
    for result in sorted_results:
        document_dict = eval(result[0], {"nan": float('nan')})  # Define 'nan' as float('nan')
        if document_dict.get('role') == '자녀':
            filtered_results.append(result)

    if not filtered_results:
        return "No results found."

    string = filtered_results[0][0]

    # 'nan'을 None으로 변경
    string = string.replace('nan', 'None')

    # 문자열을 딕셔너리로 변환
    data = ast.literal_eval(string)

    df_text = df[df['id'] == data['id']]

    if df_text.empty:
        return "No matching text found in the CSV."

    situation = '상황 부분은' + f"'{df_text.iloc[0]['situation']}'" + '이런 상황이야.'

    context_text = ''
    for index in range(len(df_text)):
        role = df_text.iloc[index]['role']
        text = df_text.iloc[index]['text']

        row = role + ':'+ text + '\n'
        context_text += row

    total_text = situation + context_text
    return total_text
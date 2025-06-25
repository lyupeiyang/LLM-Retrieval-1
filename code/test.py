import numpy as np
import pandas as pd
from ir_models.dense import Contriever
import ir_datasets
import os


def encode_and_save_documents(doc_texts, doc_ids, memmap_path, csv_map_path):
    # 实例化模型
    model = Contriever(model_hgf='facebook/contriever-msmarco')

    print(" 开始编码文档，总量：", len(doc_texts))

    # 编码
    document_embeddings = model.encode_documents(doc_texts)

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(memmap_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_map_path), exist_ok=True)

    # 保存 memmap 文件
    embedding_dim = len(document_embeddings[0])
    memmap = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(len(document_embeddings), embedding_dim))

    for idx, encoding in enumerate(document_embeddings):
        memmap[idx] = encoding

    # 保存 CSV 映射表
    offsets = np.arange(len(doc_ids))
    df = pd.DataFrame({"doc_id": doc_ids, "offset": offsets})
    df.to_csv(csv_map_path, index=False)

    print(f" 编码完成！已保存 memmap 至：{memmap_path}")
    print(f" 映射表已保存至：{csv_map_path}")


if __name__ == "__main__":
    # 加载完整 MS MARCO passage 文档集
    dataset = ir_datasets.load("msmarco-passage")

    doc_ids = []
    doc_texts = []

    print(" 开始读取文档内容...")

    #只读取前 10 条文档
    for i, doc in enumerate(dataset.docs_iter()):
        if i >= 10:
            break
        doc_ids.append(doc.doc_id)
        doc_texts.append(doc.text)

    print(f" 已读取文档数：{len(doc_ids)}")

    #print(" 开始读取文档内容...")

    #for doc in dataset.docs_iter():
        #doc_ids.append(doc.doc_id)
        #doc_texts.append(doc.text)

    #print(f" 总文档数：{len(doc_ids)}")
    # 输出路径（可根据项目结构调整）
    memmap_path = "data/memmap/msmarco-passages/Contriever/Contriever.dat"
    csv_map_path = "data/memmap/msmarco-passages/Contriever/Contriever_map.csv"

    encode_and_save_documents(doc_texts, doc_ids, memmap_path, csv_map_path)

    print("🎉 All done.")

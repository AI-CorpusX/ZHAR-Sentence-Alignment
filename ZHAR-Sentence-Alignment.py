import re
import csv
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer


# 路径配置区
# 中文txt文件绝对路径
ZH_TXT_PATH = r"修改自定义的路径位置"
# 阿语txt文件绝对路径            
AR_TXT_PATH = r"修改自定义的路径位置"
# 生成的csv文件的导出绝对路径
OUTPUT_CSV_PATH = r"修改自定义的路径位置"


# 参数配置区
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SCORE = 0.55


# CAMeL 阿语分词器
CAMEL_DISAMBIG_MODEL = "calima-msa-r13"
CAMEL_SCHEME = "d3tok"
CAMEL_SPLIT = True
CAMEL_DIAC = False


# 文本读取
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# 分句
def split_chinese_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?<=[。！？；!?;])\s*|\n+", text)
    return [s.strip() for s in parts if s.strip()]


def split_arabic_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?<=[.!؟!؛;])\s*|\n+", text)
    return [s.strip() for s in parts if s.strip()]


# CAMeL 阿语形态分词
def build_camel_tokenizer() -> MorphologicalTokenizer:
    mle = MLEDisambiguator.pretrained(CAMEL_DISAMBIG_MODEL)

    tokenizer = MorphologicalTokenizer(
        disambiguator=mle,
        scheme=CAMEL_SCHEME,
        split=CAMEL_SPLIT,
        diac=CAMEL_DIAC
    )

    return tokenizer


def camel_segment_arabic(sentence: str, tokenizer: MorphologicalTokenizer) -> str:
    words = simple_word_tokenize(sentence)
    tokens = tokenizer.tokenize(words)
    return " ".join(tokens)



# 对齐算法
def monotonic_align(sim_matrix: np.ndarray, min_score: float) -> list[tuple[int, int, float]]:
    n, m = sim_matrix.shape

    dp = np.zeros((n + 1, m + 1), dtype=float)
    back = np.empty((n + 1, m + 1), dtype=object)

    for i in range(1, n + 1):
        back[i][0] = "up"

    for j in range(1, m + 1):
        back[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = sim_matrix[i - 1][j - 1]

            best_score = dp[i - 1][j]
            best_move = "up"

            if dp[i][j - 1] > best_score:
                best_score = dp[i][j - 1]
                best_move = "left"

            if score >= min_score:
                match_score = dp[i - 1][j - 1] + score
                if match_score > best_score:
                    best_score = match_score
                    best_move = "diag"

            dp[i][j] = best_score
            back[i][j] = best_move

    alignments = []
    i, j = n, m

    while i > 0 or j > 0:
        move = back[i][j]

        if move == "diag":
            score = float(sim_matrix[i - 1][j - 1])
            alignments.append((i - 1, j - 1, score))
            i -= 1
            j -= 1
        elif move == "up":
            i -= 1
        elif move == "left":
            j -= 1
        else:
            break

    alignments.reverse()
    return alignments


# 主流程
def main():
    print("读取 TXT 文件...")

    zh_text = read_txt(ZH_TXT_PATH)
    ar_text = read_txt(AR_TXT_PATH)

    print("中文分句...")
    zh_sentences = split_chinese_sentences(zh_text)

    print("阿语分句...")
    ar_sentences = split_arabic_sentences(ar_text)

    print(f"中文句子数：{len(zh_sentences)}")
    print(f"阿语句子数：{len(ar_sentences)}")

    print("初始化 CAMeL Tools 阿语形态分词器...")
    camel_tokenizer = build_camel_tokenizer()

    print("阿语 CAMeL 分词...")
    ar_segmented_sentences = [
        camel_segment_arabic(sent, camel_tokenizer)
        for sent in ar_sentences
    ]

    print("加载多语言句向量模型...")
    model = SentenceTransformer(MODEL_NAME)

    print("生成中文句向量...")
    zh_embeddings = model.encode(
        zh_sentences,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("生成阿语句向量，使用 CAMeL 分词后的文本...")
    ar_embeddings = model.encode(
        ar_segmented_sentences,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("计算相似度矩阵...")
    sim_matrix = np.matmul(zh_embeddings, ar_embeddings.T)

    print("执行中阿单调对齐...")
    alignments = monotonic_align(sim_matrix, MIN_SCORE)

    print(f"保留对齐结果数：{len(alignments)}")

    rows = []

    for align_id, (zh_idx, ar_idx, score) in enumerate(alignments, start=1):
        rows.append({
            "align_id": align_id,
            "zh_index": zh_idx,
            "ar_index": ar_idx,
            "zh_text": zh_sentences[zh_idx],
            "ar_text": ar_sentences[ar_idx],
            "ar_camel_tokens": ar_segmented_sentences[ar_idx],
            "score": round(score, 6)
        })

    df = pd.DataFrame(rows)

    print("导出 CSV...")
    df.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL
    )

    print(f"完成。CSV 已保存到：{OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
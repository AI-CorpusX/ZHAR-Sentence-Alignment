# 中阿句子对齐工具（Chinese–Arabic Sentence Alignment Tool）
<img width="912" height="549" alt="image" src="https://github.com/user-attachments/assets/36da7560-6066-4aff-9e0b-99d6beb8e960" />

[AI CprpusX课题组官网链接](https://aicorpusx.com)

---
基于：

- CAMeL Tools 阿语形态分词
- multilingual SentenceTransformer 多语言句向量
- 单调动态规划（Monotonic Dynamic Programming）

---

## 功能特性

- 中文 / 阿语 TXT 文本读取
- 自动分句
- CAMeL Tools 阿语形态分词
- 多语言句向量生成
- 基于余弦相似度的跨语言匹配
- 单调动态规划路径约束
- CSV 对齐结果导出

---

## 项目结构

```text
ZhArSentAlign/
│
├── ZHAR-Sentence-Alignment.py
└── readme
```

---

## 安装依赖

```bash
pip install camel-tools sentence-transformers pandas numpy scikit-learn
```

---

## 安装 CAMeL Tools 数据

首次使用需要下载阿语形态分析模型。

现代标准阿拉伯语（默认）：

```bash
camel_data -i disambig-mle-calima-msa-r13
```

埃及方言：

```bash
camel_data -i disambig-mle-calima-egy-r13
```

---

## 输入文件

修改脚本中的路径：

```python
# 中文txt文件绝对路径
ZH_TXT_PATH = r"修改自定义的路径位置"
# 阿语txt文件绝对路径            
AR_TXT_PATH = r"修改自定义的路径位置"
# 生成的csv文件的导出绝对路径
OUTPUT_CSV_PATH = r"修改自定义的路径位置"
```

---

## 使用方法

运行：

```bash
python zh_ar_align.py
```

程序流程：

1. 读取中文与阿语 TXT
2. 自动分句
3. 阿语 CAMeL 形态分词
4. 生成多语言句向量
5. 计算相似度矩阵
6. 动态规划句对齐
7. 导出 CSV

---

## 输出结果

输出 CSV 示例：

| align_id | zh_text | ar_text | score |
|---|---|---|---|
| 1 | 我喜欢学习。 | أحب التعلم | 0.82 |

额外字段：

- zh_index
- ar_index
- ar_camel_tokens
- score

---

## 模型选择

默认模型：

```python
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

特点：
- 速度快
- 占用小
- 适合一般中阿对齐任务

可替换为：

```python
sentence-transformers/LaBSE
```

特点：
- 更适合跨语言对齐
- 平行语料效果更好
- 更慢、更占内存

---

## 参数说明

最低对齐阈值：

```python
MIN_SCORE = 0.55
```

建议：

| 情况 | 调整 |
|---|---|
| 漏对齐较多 | 调低 |
| 错对齐较多 | 调高 |

推荐范围：

```python
0.45 ~ 0.65
```

---

## CAMeL 配置

推荐默认配置：

```python
CAMEL_DISAMBIG_MODEL = "calima-msa-r13"
CAMEL_SCHEME = "d3tok"
CAMEL_SPLIT = True
CAMEL_DIAC = False
```

具体说明书如下：
1. 消歧模型 disambiguator：
calima-msa-r13
用于现代标准阿拉伯语 MSA，适合：新闻、正式文章、书面语、政府文档、教材文本。推荐中阿正式文档对齐时使用这个。
calima-egy-r13
用于埃及阿拉伯语。适合：埃及方言、口语化文本、社交媒体文本。注意官方示例中埃及方言资源主要搭配 bwtok。

2. 分词方案 scheme：
d3tok
较常用，适合中阿句对齐。会拆分常见黏着成分，例如：و + ب + ال + مدرسة ，推荐正式阿语、新闻、普通中阿文档对齐。
atbtok
相对保守，不会像 bwtok 那么细。不想切得太碎时使用。
bwtok
切分更细。词法分析、语言学分析、需要细粒度形态信息时使用。用于句向量对齐时可能过细，不一定总是提升。

3. split 参数：
split=True
输出为拆开的 token 列表。
例如可能输出：
["و", "ب", "ال", "مدرسة"]
推荐用于后续 embedding。split=False
输出为带连接符的形式。
例如可能输出：
"و+_ب+_ال+_مدرسة"
更适合观察分词结果，不一定适合 embedding。

4. diac 参数：
diac=False
不保留阿语元音符号。
推荐用于中阿对齐，因为大多数现代阿语文本本来就没有元音符号。
diac=True
保留元音符号。
适合古典阿语、教学文本、需要读音信息的任务。
但用于普通 embedding 对齐时，可能增加噪声。

---

## 分句说明

```python
def split_chinese_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?<=[。！？；!?;])\s*|\n+", text)
    return [s.strip() for s in parts if s.strip()]


def split_arabic_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?<=[.!؟!؛;])\s*|\n+", text)
    return [s.strip() for s in parts if s.strip()] 
```

- 中文分句默认按照 。！？； 和换行切分。
- 阿语分句默认按照 . ؟ ! ؛ 和换行切分。
- 如果有特殊的要求记得改切分符号，按照自己的需求进行修改

---

## 适用场景

- 中阿平行语料构建
- 机器翻译数据预处理
- NLP 研究
- 双语文本整理
- 翻译辅助

---

## 作者

Feng Yifan

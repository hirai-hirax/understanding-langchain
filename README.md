# Understanding LangChain - 最小限のQ&Aシステム

LangChainを使って、特定のフォルダ内のドキュメントに基づいて質問に回答するシンプルなシステムです。

## 必要な準備

### 1. 環境変数の設定
`.env`ファイルを作成し、OpenAI APIキーを設定してください：
```
OPENAI_API_KEY=your-api-key-here
```

### 2. 依存関係のインストール
```bash
uv sync
```

## 使い方

### 基本的な実行方法
```bash
uv run python main.py
```

実行すると以下のプロンプトが表示されます：
1. `Folder:` - ドキュメントが含まれるフォルダのパスを入力
2. `Question:` - 質問を入力

### テスト用ドキュメントでの実行例
```bash
# テストドキュメントフォルダを指定
Folder: test_docs
Question: Pythonの開発者は誰ですか？
```

## コードの詳細解説

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def qa_system(folder: str, question: str) -> str:
    docs = DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
    return RetrievalQA.from_chain_type(ChatOpenAI(temperature=0), retriever=vectorstore.as_retriever()).invoke({"query": question})["result"]

if __name__ == "__main__":
    print(qa_system(input("Folder: "), input("Question: ")))
```

### 1行目から6行目：必要なモジュールのインポート
```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
```
- `OpenAIEmbeddings`: テキストをベクトル（数値の配列）に変換するOpenAIの埋め込みモデル
- `ChatOpenAI`: OpenAIのチャットモデル（GPT-4o、GPT-4o-mini等）を使うためのクラス

```python
from langchain_community.vectorstores import Chroma
```
- `Chroma`: ベクトルデータベース。後述の詳細解説を参照

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
```
- `DirectoryLoader`: フォルダ内のファイルを一括で読み込むローダー
- `TextLoader`: 個々のテキストファイルを読み込むローダー

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```
- `RecursiveCharacterTextSplitter`: 長い文書を小さなチャンクに分割するツール

```python
from langchain.chains import RetrievalQA
```
- `RetrievalQA`: 検索ベースの質問応答を行うチェーン（処理の連鎖）

```python
from dotenv import load_dotenv
```
- `load_dotenv`: .envファイルから環境変数を読み込む

### Chromaベクトルデータベースの詳細

Chromaは、テキストの「意味」を数値化して保存・検索するためのデータベースです。

#### なぜベクトルデータベースが必要か？
通常のキーワード検索では「Python開発者」と「Pythonを作った人」が同じ意味だと認識できません。ベクトル化により、意味的に近い内容を見つけることができます。

#### Chromaの動作原理
1. **埋め込み（Embedding）**: テキストを高次元ベクトル（通常1536次元）に変換
   - 例：「Pythonは1991年に開発された」→ [0.12, -0.34, 0.56, ...]
2. **保存**: これらのベクトルをChromaのインメモリデータベースに保存
3. **検索**: 質問文もベクトル化し、コサイン類似度で最も近いベクトルを検索
4. **取得**: 類似度の高い元のテキストチャンクを取得

#### Chromaの利点
- **高速**: インメモリで動作するため検索が高速
- **簡単**: 複雑な設定不要で即座に使用可能
- **軽量**: 外部サーバー不要でローカル実行可能

### 8行目：環境変数の読み込み
```python
load_dotenv()
```
.envファイルからOPENAI_API_KEYなどの環境変数を読み込みます。

### 10行目：メイン関数の定義
```python
def qa_system(folder: str, question: str) -> str:
```
フォルダパスと質問を受け取り、回答を返す関数を定義します。

### 11行目：ドキュメントの読み込み
```python
docs = DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}).load()
```
この行で行われる処理：
1. `DirectoryLoader`で指定フォルダ内のローダーを作成
2. `glob="**/*.txt"`で全サブフォルダの.txtファイルを対象に指定
3. `loader_cls=TextLoader`で各ファイルをTextLoaderで読み込むよう指定
4. `loader_kwargs={'encoding': 'utf-8'}`でUTF-8エンコーディングを指定（日本語対応）
5. `.load()`で実際にファイルを読み込み、ドキュメントリストを取得

### 12行目：テキストの分割
```python
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
```
この行で行われる処理：
1. `RecursiveCharacterTextSplitter`でテキスト分割器を作成
2. `chunk_size=500`で各チャンクを最大500文字に設定
3. `chunk_overlap=50`で隣接チャンク間で50文字重複させる（文脈の連続性を保つため）
4. `.split_documents(docs)`で実際にドキュメントを分割

### 13行目：ベクトルストアの作成
```python
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
```
この行で行われる処理：
1. `OpenAIEmbeddings()`でOpenAIの埋め込みモデル（text-embedding-ada-002）を初期化
2. 各チャンクのテキストを1536次元のベクトルに変換
3. `Chroma.from_documents()`でベクトルとメタデータをChromaデータベースに保存
4. 質問時に意味的に類似したチャンクを高速検索できる状態にする

### 14行目：質問応答の実行
```python
return RetrievalQA.from_chain_type(ChatOpenAI(temperature=0), retriever=vectorstore.as_retriever()).invoke({"query": question})["result"]
```
この行で行われる処理：
1. `ChatOpenAI(temperature=0)`でOpenAIのチャットモデルを初期化（デフォルトはgpt-3.5-turbo、temperature=0で確定的な回答）
2. `vectorstore.as_retriever()`でベクトルストアから検索器を作成
3. `RetrievalQA.from_chain_type()`で検索ベースの質問応答チェーンを作成
4. `.invoke({"query": question})`で質問を実行
5. `["result"]`で回答部分のみを取得して返す

### 16-17行目：メイン処理
```python
if __name__ == "__main__":
    print(qa_system(input("Folder: "), input("Question: ")))
```
スクリプトが直接実行された場合：
1. `input("Folder: ")`でフォルダパスの入力を求める
2. `input("Question: ")`で質問の入力を求める
3. `qa_system()`を呼び出して回答を取得
4. `print()`で回答を表示

## 処理の流れ

1. **文書読み込み**: 指定フォルダから.txtファイルを読み込む
2. **チャンク分割**: 長い文書を500文字の小さな断片に分割
3. **ベクトル化**: 各チャンクをOpenAIの埋め込みモデルで数値ベクトルに変換
4. **ベクトル保存**: Chromaデータベースにベクトルを保存
5. **質問処理**: 質問もベクトル化し、類似度の高いチャンクを検索
6. **回答生成**: 検索結果を基にChatGPTが回答を生成

## トラブルシューティング

### エラー: `ValueError: Expected Embeddings to be non-empty list`
- 原因：ドキュメントが正しく読み込まれていない
- 解決方法：フォルダパスが正しいか、.txtファイルが存在するか確認

### エラー: `OpenAI API key not found`
- 原因：環境変数が設定されていない
- 解決方法：`.env`ファイルにOPENAI_API_KEYを設定

## カスタマイズ方法

### 対応ファイル形式を変更
```python
# PDFファイルも読み込む場合
glob="**/*.pdf"  # または "**/*.{txt,pdf}"
```

### チャンクサイズを調整
```python
# より大きなコンテキストで処理
chunk_size=1000, chunk_overlap=100
```

### 使用モデルを変更
```python
# GPT-4o-miniを明示的に指定（デフォルト）
ChatOpenAI(model="gpt-4o-mini", temperature=0)

# GPT-4oを使用（より高精度）
ChatOpenAI(model="gpt-4o", temperature=0)

```

## LangChainありとなしの比較

このプロジェクトには、同じ機能を実装した2つのファイルが含まれています：

### main.py（LangChain使用）
LangChainフレームワークを使用した実装です。簡潔で読みやすく、高レベルなAPIを活用しています。

### main2.py（LangChainなし）
LangChainを使わずにOpenAI APIを直接使用した実装です。同じ機能を手動で実装することで、LangChainが内部的に何を行っているかを理解できます。

#### main2.pyの実装内容：
- **ファイル読み込み**: `glob`と`pathlib`を使用してフォルダ内の.txtファイルを手動で読み込み
- **テキスト分割**: 単純なスライスを使用してテキストを500文字のチャンクに分割（50文字のオーバーラップ）
- **埋め込み生成**: OpenAI APIを直接呼び出してテキストの埋め込みベクトルを生成
- **類似度計算**: NumPyを使用してコサイン類似度を手動で計算
- **コンテキスト選択**: 類似度の高い上位3つのチャンクを選択
- **回答生成**: OpenAI Chat APIを直接呼び出して回答を生成

#### 比較して学べること：
1. **抽象化の価値**: LangChainがどれだけ複雑な処理を隠蔽しているか
2. **実装の複雑さ**: 手動実装では埋め込み、類似度計算、コンテキスト管理などを自分で行う必要
3. **エラーハンドリング**: LangChainは内部的にエラーハンドリングを行っているが、手動実装では自分で処理する必要
4. **性能**: 手動実装では最適化の余地があるが、LangChainは一般的なユースケースに最適化済み

### 実行方法
両方のファイルは同じインターフェースで実行できます：

```bash
# LangChain版
uv run python main.py

# LangChainなし版
uv run python main2.py
```
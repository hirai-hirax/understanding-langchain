import os
from main import qa_system

# Set test environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-key")

# Test the Q&A system
if __name__ == "__main__":
    test_folder = "./test_docs"
    test_questions = [
        "Pythonを開発したのは誰ですか？",
        "LangChainの主要コンポーネントは何ですか？",
        "Pythonのデータ型を教えてください"
    ]

    print("Testing Q&A System\n" + "="*50)

    for question in test_questions:
        print(f"\n質問: {question}")
        try:
            answer = qa_system(test_folder, question)
            print(f"回答: {answer}")
        except Exception as e:
            print(f"エラー: {e}")

    print("\n" + "="*50 + "\nTest completed")
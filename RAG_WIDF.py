import os
import streamlit as st
import snowflake.snowpark as snowpark
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Snowflake Streamlit 메인 함수
def main(session: snowpark.Session):
    st.set_page_config(page_title="설계관리자료 RAG 챗봇", page_icon="💡")
    st.title("📝 설계관리자료 기반 RAG 챗봇")

    # 1️⃣ Snowflake Secret에서 OpenAI API 키 불러오기
    try:
        secret = session.get_secret("openai_api_key")
        openai_key = secret["secret_string"]
        os.environ["OPENAI_API_KEY"] = openai_key
        st.success("✅ OpenAI API Key가 Snowflake Secret에서 성공적으로 로드되었습니다.")
    except Exception as e:
        st.error(f"❌ OpenAI API Key 불러오기 실패: {e}")
        st.stop()

# 🔹 Streamlit 캐시 사용: VectorStore를 캐시하여 반복 로딩 방지
@st.cache_resource(show_spinner=False)
def load_vectorstore(pdf_path="test manual.pdf"):
    # ✅ 1. PDF 로더 교체: PyPDFLoader → PDFPlumberLoader (더 안정적)
    try:
        loader = PDFPlumberLoader(pdf_path)
    except Exception:
        loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    # ✅ 2. 문서 페이지 수 확인
    st.write(f"📄 PDF에서 추출된 페이지 수: {len(documents)}")

    # ✅ 3. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 더 크게 (GPU/CPU 부담 줄이기)
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)
    st.write(f"🔹 분할된 청크 개수: {len(docs)}")

    # ✅ 4. Embedding 및 Vectorstore 생성
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def main():
    st.set_page_config(page_title="설계관리자료 RAG 챗봇", page_icon="💡")
    st.title("📝 설계관리자료 기반 RAG 챗봇")

    pdf_path = st.text_input("관련자료:", "test manual.pdf")

    if not os.path.isfile(pdf_path):
        st.warning("❌ PDF 파일을 찾을 수 없습니다.")
        return

    with st.spinner("VectorStore 로드 중..."):
        db = load_vectorstore(pdf_path)

    # Retriever 설정
    retriever = db.as_retriever(search_kwargs={"k": 5})

    chat_model = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    query = st.text_input("질문을 입력하세요:")
    if query:
        with st.spinner("답변 생성 중..."):
            result = qa_chain(query)
        st.markdown("### 💡 답변")
        st.write(result["result"])

        st.markdown("### 📚 관련근거")
        for i, doc in enumerate(result["source_documents"], 1):
            st.write(f"--- 참고내용 {i} ---")
            st.write(doc.page_content)

if __name__ == "__main__":

    main()








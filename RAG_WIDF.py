mport os
import streamlit as st
import snowflake.snowpark as snowpark
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Streamlit 캐시: VectorStore는 최초 한 번만 생성
@st.cache_resource(show_spinner=False)
def load_vectorstore(pdf_path="test manual.pdf"):
    try:
        loader = PDFPlumberLoader(pdf_path)
    except Exception:
        loader = PyPDFLoader(pdf_path)

    documents = loader.load()
    st.write(f"📄 PDF 페이지 수: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)
    st.write(f"🔹 분할된 청크 수: {len(docs)}")

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# ✅ Snowflake Streamlit entry point
def main(session: snowpark.Session):
    st.set_page_config(page_title="설계관리자료 RAG 챗봇", page_icon="💡")
    st.title("📝 설계관리자료 기반 RAG 챗봇")

    # 1️⃣ OpenAI API 키 불러오기
    try:
        openai_key = session.get_secret("openai_api_key")
        os.environ["OPENAI_API_KEY"] = openai_key
        st.success("✅ OpenAI API Key가 성공적으로 로드되었습니다.")
    except Exception as e:
        st.error(f"❌ API Key 로드 실패: {e}")
        st.stop()

    # 2️⃣ PDF 경로 설정 (Snowflake에서는 로컬 접근 불가)
    pdf_path = "test manual.pdf"

    # 3️⃣ VectorStore 로드
    with st.spinner("VectorStore 로딩 중..."):
        db = load_vectorstore(pdf_path)

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

        st.markdown("### 📚 참고 문서")
        for i, doc in enumerate(result["source_documents"], 1):
            st.write(f"--- 문서 {i} ---")
            st.write(doc.page_content)

import os
import streamlit as st
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# LangChain & OpenAI Imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Sistema Literário Inteligente",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# CONEXÃO COM O BANCO DE DADOS (PostgreSQL)
# ==========================================
DATABASE_URL = st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL", ""))

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

@st.cache_resource
def get_db_engine():
    return create_engine(
        DATABASE_URL,
        connect_args={"sslmode": "require"} if "sslmode" not in DATABASE_URL else {}
    )

engine = get_db_engine()
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Modelos SQLAlchemy simplificados
class LivroModel(Base):
    __tablename__ = "livros"
    id = Column(Integer, primary_key=True)
    titulo = Column(String)
    subtitulo = Column(String)
    genero_literario = Column(String)
    status = Column(String)

class PersonagemModel(Base):
    __tablename__ = "personagens"
    id = Column(Integer, primary_key=True)
    livro_id = Column(Integer, ForeignKey("livros.id"))
    nome_completo = Column(String)
    idade = Column(Integer)
    personalidade = Column(Text)
    motivacoes = Column(Text)
    conflitos_internos = Column(Text)

class CapituloModel(Base):
    __tablename__ = "capitulos"
    id = Column(Integer, primary_key=True)
    livro_id = Column(Integer, ForeignKey("livros.id"))
    numero = Column(Integer)
    titulo = Column(String)
    quantidade_palavras = Column(Integer)
    status_escrita = Column(String)

# ==========================================
# FERRAMENTAS DA IA (TOOLS)
# ==========================================
@tool
def buscar_ficha_personagem(livro_id: int, nome_personagem: str) -> str:
    """Busca a ficha detalhada de um personagem no banco de dados do livro."""
    db = SessionLocal()
    try:
        p = db.query(PersonagemModel).filter(
            PersonagemModel.livro_id == livro_id,
            PersonagemModel.nome_completo.ilike(f"%{nome_personagem}%")
        ).first()
        if not p: return f"Personagem '{nome_personagem}' não encontrado."
        return f"Nome: {p.nome_completo}\nIdade: {p.idade}\nPersonalidade: {p.personalidade}\nMotivações: {p.motivacoes}\nConflitos: {p.conflitos_internos}"
    finally:
        db.close()

tools = [buscar_ficha_personagem]

# ==========================================
# INTERFACE GRÁFICA (STREAMLIT)
# ==========================================
st.title("📚 Central de Desenvolvimento Literário")
st.markdown("Gerencie seu universo narrativo e consulte seu co-piloto de IA em tempo real.")

# Sidebar para seleção de Livro
st.sidebar.header("Configurações do Projeto")
livro_id_input = st.sidebar.number_input("ID do Livro Ativo", min_value=1, value=1, step=1)

# Abas principais da aplicação
aba_dashboard, aba_personagens, aba_capitulos, aba_chat = st.tabs([
    "📊 Dashboard", "👥 Personagens", "📖 Capítulos", "🤖 Co-piloto IA"
])

db = SessionLocal()

with aba_dashboard:
    st.subheader("Métricas Gerais da Obra")
    try:
        livro = db.query(LivroModel).filter(LivroModel.id == livro_id_input).first()
        
        if livro:
            caps = db.query(CapituloModel).filter(CapituloModel.livro_id == livro_id_input).all()
            total_palavras = sum(c.quantidade_palavras for c in caps if c.quantidade_palavras)
            total_caps = len(caps)
            concluidos = sum(1 for c in caps if c.status_escrita == 'Concluído')
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Título do Livro", livro.titulo)
            col2.metric("Total de Capítulos", f"{total_caps} ({concluidos} prontas)")
            col3.metric("Total de Palavras", f"{total_palavras:,}")
            col4.metric("Status do Projeto", livro.status)
        else:
            st.warning(f"Nenhum livro encontrado com o ID {livro_id_input}. Verifique o banco de dados.")
    except Exception as e:
        st.error(f"Erro ao carregar o dashboard: {e}")

with aba_personagens:
    st.subheader("Elenco de Personagens")
    try:
        df_personagens = pd.read_sql(
            f"SELECT nome_completo, idade, personalidade, motivacoes FROM personagens WHERE livro_id = {livro_id_input}",
            engine
        )
        if not df_personagens.empty:
            st.dataframe(df_personagens, use_container_width=True)
        else:
            st.info("Nenhum personagem cadastrado para este livro ainda.")
    except Exception as e:
        st.error(f"Erro ao carregar personagens: {e}")

with aba_capitulos:
    st.subheader("Estrutura de Capítulos")
    try:
        df_caps = pd.read_sql(
            f"SELECT numero, titulo, quantidade_palavras, status_escrita FROM capitulos WHERE livro_id = {livro_id_input} ORDER BY numero",
            engine
        )
        if not df_caps.empty:
            st.dataframe(df_caps, use_container_width=True)
        else:
            st.info("Nenhum capítulo cadastrado para este livro ainda.")
    except Exception as e:
        st.error(f"Erro ao carregar capítulos: {e}")

with aba_chat:
    st.subheader("💬 Chat com o Co-piloto Editorial (IA)")
    st.markdown("Converse com a IA sobre o enredo, inconsistências ou detalhes dos personagens.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ex: Qual é o conflito interno do personagem principal?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando o banco de dados e estruturando resposta..."):
                try:
                    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
                    llm_with_tools = llm.bind_tools(tools)
                    
                    system_msg = SystemMessage(content=(
                        f"Você é o Co-piloto Editorial do livro ID {livro_id_input}. "
                        "Use as ferramentas para consultar dados reais se necessário."
                    ))
                    
                    messages = [system_msg, HumanMessage(content=prompt)]
                    response = llm_with_tools.invoke(messages)
                    
                    if response.tool_calls:
                        for tool_call in response.tool_calls:
                            tool_args = tool_call["args"]
                            tool_args["livro_id"] = livro_id_input
                            tool_output = buscar_ficha_personagem.invoke(tool_args)
                            messages.append(HumanMessage(content=f"[Dados do Banco]: {tool_output}"))
                        
                        final_res = llm.invoke(messages)
                        resposta_final = final_res.content
                    else:
                        resposta_final = response.content
                        
                    st.markdown(resposta_final)
                    st.session_state.messages.append({"role": "assistant", "content": resposta_final})
                except Exception as e:
                    st.error(f"Erro ao comunicar com a IA: {e}")

db.close()

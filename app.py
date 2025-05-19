
chiave = st.secrets["superkey"]

import streamlit as st

from PIL import Image
logo = Image.open("Chatbot.webp")
st.image(logo)

with st.sidebar: #with significa prendi in considerazione che stiamo scrivendo questa cosa in una sidebar e i due punti indica un indetnazione che indica che le cose che scriveremo andranno nella barra laterale.
  file = st.file_uploader("Carica il tuo file", type ="pdf") #mettiamo un file uploader dove puoi caricaricare il tuo file che deve essere in pdf

from PyPDF2 import PdfReader #libreria per la gestione dei pdf

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS                              #si importano molte cose da langchain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

if file is not None:  #se il file non è vuoto fai le instruzioni che arrivano dopo
    testo_letto = PdfReader(file) #testo_letto legge il file che è stato caricato

    testo = "" #creiamo una variabile testo che all'inizio e vuota e succevamente andiamo a rimpire con i pezzi di testo
    for pagina in testo_letto.pages: #per ogni pagina dentro testo letto
        testo = testo + pagina.extract_text() #il testo sarà uguale al testo precedente + estrai testo dalla pagina 1, poi 2 ecc... alla fine avremo il contenitore testo dove avremp tutte le pagine
        # st.write(testo)

    # Usiamo il text splitter di Langchain
    testo_spezzato = RecursiveCharacterTextSplitter(
        separators="\n", #il criterio per separare un testo dall'altro è l'andare a capo
        chunk_size=1000, # Numero di caratteri per chunk, viene spezzato in pezzi da 1000 caratteri
        chunk_overlap=150, #può capitare che tagliando un testo il discorso non si finito nonostante il testo vada a capo, questo paramentro cerca di evitare questo errore sovrapponendo 150 caratteri tra un taglio e un altro
        length_function=len
        )

    pezzi = testo_spezzato.split_text(testo) #prendi testo spezzato e lo tagli, la vera operazione di taglio.
    #st.write(pezzi)

    # Generazione embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=chiave) #questa variabile andrà a prendere tutti gli embeddings di openAI

    # Vector store - FAISS (by Facebook)
    vector_store = FAISS.from_texts(pezzi, embeddings) #FAISS ci da i vector score e andiamo a utilizzare quelli di meta, nel vector score andranno tutti gli embeddings dei pezzi

    # Prompt
    domanda = st.text_input("Chiedi al chatbot:") #con questa funzione di strimlit possiamo mettere il testo in iput dove si possiamo chiedere al chatbot

    if domanda:
        st.write("Sto cercado le informazioni che mi hai richiesto...")
        rilevanti = vector_store.similarity_search(domanda)

        # Definiamo l'LLM
        llm = ChatOpenAI(
            openai_api_key = chiave,
            temperature = 1.0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo-0125")
        # https://platform.openai.com/docs/models/compare

        # Output
        # Chain: prendi la domanda, individua i frammenti rilevanti,
        # passali all'LLM, genera la risposta
        chain = load_qa_chain(llm, chain_type="stuff")
        risposta = chain.run(input_documents = rilevanti, question = domanda)
        st.write(risposta)


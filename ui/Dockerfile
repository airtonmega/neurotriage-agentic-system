FROM python:3.12-slim

WORKDIR /app

# Instalar dependências
RUN pip install --no-cache-dir \
    streamlit==1.31.0 \
    requests==2.31.0 \
    streamlit-lottie==0.0.5

# Copiar app
COPY app.py .

# Streamlit roda por padrão na porta 8501, mas Cloud Run espera $PORT
# Vamos configurar o Streamlit para usar a variável $PORT
ENV PORT=8080

EXPOSE ${PORT}

CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0

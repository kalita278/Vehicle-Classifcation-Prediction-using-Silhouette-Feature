FROM python:3.8
COPY . /app
WORKDIR /app
ENV streamlit_run_host = 0.0.0.0
RUN pip install -r requirements.txt
CMD streamlit run "prediction_web_app.py"

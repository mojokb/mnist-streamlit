FROM dudaji/cap-jupyterlab:tf2.3-cpu

WORKDIR /app

ADD app.py /app
ADD train.ipynb /app

CMD ["streamlit", "run",  "app.py", "--serverport=80"]
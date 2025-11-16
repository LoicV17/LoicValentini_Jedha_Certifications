FROM apache/airflow:2.7.3

# Copier ton requirements.txt dans l'image
COPY requirements.txt /requirements.txt

# Installer les libs Python
RUN pip install --no-cache-dir -r /requirements.txt

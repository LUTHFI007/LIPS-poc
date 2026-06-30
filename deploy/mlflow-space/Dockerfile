FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Pin MLflow to the SAME version as the app's client (the `lips` env, 3.1.4) to
# avoid any schema/version mismatch. psycopg2-binary is the Postgres driver.
RUN pip install --no-cache-dir mlflow==3.1.4 psycopg2-binary==2.9.9

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 7860

CMD ["/start.sh"]

FROM python:3.10-slim

WORKDIR /app

# Copy SDK (from build context, staged by build_app.sh)
COPY hailo_ipc_sdk/ /app/hailo_ipc_sdk/
COPY setup.py README.md /app/
RUN pip install --no-cache-dir -e . && rm -rf hailo_ipc_sdk setup.py README.md

COPY app.py .

CMD ["python", "-u", "app.py"]

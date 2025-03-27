FROM ubuntu:22.04 AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y python3.11 python3-pip
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

FROM ubuntu:22.04
WORKDIR /app
RUN apt-get update && apt-get install -y uwsgi uwsgi-plugin-python3 libgl1-mesa-glx
COPY --from=builder /usr/local/lib/python3.11/dist-packages/ /usr/local/lib/python3.11/dist-packages/
COPY . .
CMD ["uwsgi", "--ini", "uwsgi.ini"]
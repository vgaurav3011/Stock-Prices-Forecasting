FROM ubuntu:18.04

# ubuntu installing - python, pip, graphviz
RUN apt-get update &&\
    apt-get install python3.8 -y &&\
    apt-get install python3-pip -y &&\
    apt-get install graphviz -y
#  Streamlit Default Port
EXPOSE 8501
# Directory creation for the streamlit app
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD streamlit run app.py

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

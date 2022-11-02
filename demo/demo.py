from operator import concat
import streamlit as st
import requests
import yaml
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

with open("demo.yaml", "r") as stream:
    try:
        env_vars = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

def main():
    request_url = env_vars['components'][0]['env']['INFER_URL']
    st.set_page_config(page_title="Twitter Sentiment Analysis")
    add_bg_from_local('twitter1.jpeg')
    st.title("Twitter Sentiment Analysis")
    st.header('Welcome to Twitter Sentiment Analysis!')
    st.write('This is a sample app that demonstrates the prowess of ServiceFoundry ML model deployment.🚀')
    st.write('Visit the [Github](https://github.com/urja0901/twitter-sentiment-analysis) repo for code or [Google Colab](https://colab.research.google.com/drive/1hwIC356LfLaWMPS4xYFRB93XxCF18KSb#scrollTo=bTwryoAy4I9s) notebook for a quick start.')
    with st.form("my_form"):
        
        sentiment_text = st.text_input('Sentiment Text',value="It's a good day!")

        features = {
                "tweet": sentiment_text
            }
            
        submitted = st.form_submit_button("Submit")
        if submitted:
            data = requests.post(url=concat(request_url, "/predict"), params=features).json()
            if data:
                if data["sentiment"] == 0 :
                    return_val = "positive"
                else : 
                    return_val = "negative (racist/sexist)"
                st.metric(label="sentiment",value=data['sentiment'])
            else:
                st.error("Error")

    st.image('twitter2.jpeg', use_column_width='always')

if __name__ == '__main__':
    main()
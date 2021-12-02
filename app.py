import faiss
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import requests 
import io


@st.cache
def read_data(s3_data='stocks_crypto_all.csv'):
    """Read the data."""
    # return pd.read_csv(s3_data)
    header = {
    "X-API-Key": "c0u91dyk_viAjdWRh6vz9TVMFNKQca5cu7FfHqAfd",
    "Content-Type": "application/json"
    }

    detadrive = 'https://drive.deta.sh/v1/c0u91dyk/stocks/files/download?name=stocks_crypto_all.csv'

    data = requests.get(detadrive, headers=header)
    s = data.content
    return pd.read_csv(io.StringIO(s.decode('utf-8')))
    

@st.cache(allow_output_mutation=True)
def load_bert_model(name="all-mpnet-base-v2"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)

@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="stocks_crypto_indx.index"):
    """Load the Faiss index."""
    index = faiss.read_index(path_to_faiss) 
    return index


def id2details(df, I):
    return [dict(
                 title = df[df.index_id == idx]['title'],
                 description = df[df.index_id == idx]['selftext'],
                 full_link = df[df.index_id == idx]['full_link'],
                 score = df[df.index_id == idx]['score'],
                 comments = df[df.index_id == idx]['num_comments']) 
            for idx in I[0]]

def main():
    # Load data and models
    data = read_data()
    model = load_bert_model()
    index = load_faiss_index()

    st.title("Similarity search on (parts of) reddit")
    st.write('Search all posts on r/wallstreetbets, r/stocks, r/investing and r/CryptoCurrency from 2021.')
    # User search
    user_input = st.text_input("Search box", placeholder="tips on investing")
    
    if user_input:
        # Get paper IDs
        xq = model.encode([user_input])
        D, I  = index.search(xq, 5)
        


        for reddit_post in id2details(data, I):
            return_obj = {
                'title': reddit_post['title'].values[0],
                'description': reddit_post['description'].values[0],
                'link': reddit_post['full_link'].values[0],
                'score': reddit_post['score'].values[0],
                'comments': reddit_post['comments'].values[0]
            }
            # print(return_obj)
            # st.write(reddit_post['title'].values[0])

            st.subheader(return_obj['title'])
            st.write(f"upvotes/downvotes: {return_obj['score']}, comments: {return_obj['comments']}" )
            st.write(return_obj['link'])
            st.caption(return_obj['description'])
            
            st.write('/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////')
           

            
            # st.write(
            #    f"""**{reddit_post['title'].values[0]}**  
            #            {reddit_post['description'].values[0]}  
            # **Upvote/Downvote**: {reddit_post['score'].values[0]},  ** comments: {reddit_post['comments'].values[0]} ** 
            # **Reddit url**: 
            # {reddit_post['full_link'].values[0]}""") 

if __name__ == "__main__":
    main()
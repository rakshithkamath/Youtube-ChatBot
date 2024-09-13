from typing import List
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, BaseRetriever



# Function to Extract Video ID from Youtube URL
def extract_video_id(url):
    """
    Helper function used to get the youtube URL from the User's input 
    """

    try:
        if "youtu.be" in url:
            return url.split("/")[-1]
        
        elif "youtube.com" in url:
            return url.split("v=")[1].split("&")[0]
        
        else:
            raise ValueError("Invalid Youtube URL")
        
    except:
        raise ValueError("Could not extract video ID from the provided URL")
    

# Function to get video transcript
def get_transcript(video_id):
    """
    Helper function used to get the transcript of a youtube video if present using Youtube Transcript API
    """

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        result = []
        for i in transcript:
            result.append({
                'text': i['text'],
                'start': i['start'],
                'end': i['start'] + i['duration']
            })
        return result
    
    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video")
    
    except NoTranscriptFound:
        raise ValueError("No transcript found for this video")
    
    except Exception as e:
        raise ValueError(f"An error occurred while fetching the transcript: {str(e)}")
    
    
# Function to format time
def format_time(seconds):
    """
    Function used to convert the time stamps recived in the output to the right format in the final answer
    """

    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# Function to create Vector Store
def create_vector_store(transcript):
    """
    Function to create the the vector store using hugging face embeddings and using FAISS for storing the vectors. 
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([seg['text'] for seg in transcript], metadatas=[{'start': seg['start'], 'end': seg['end']} for seg in transcript])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# Custom Retriever
class CustomRetriever(BaseRetriever):
    """
    Creating a Custom Retriver class to handle the timestamp metadata while doing the retrival before passing it to the llm. 
    """
    
    vector_store: FAISS  # Declare as a class variable with type annotation since the class is Pydantic style

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types like FAISS since the default type for class is different to our use case 

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.vector_store.similarity_search(query, k=5, fetch_k=10)
        for doc in docs:
            doc.page_content += f" [Timestamp: {doc.metadata['start']:.2f} - {doc.metadata['end']:.2f}]"
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

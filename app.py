import re
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from utils import extract_video_id,get_transcript,format_time,create_vector_store,CustomRetriever




# Video Summarizer 
def create_video_summary(transcript_text):
    """
    Creating a function to create summary of the video usiing multi-steps for longer videos. 
    This is used in the prompt template along with prompt to help aid the llm to answer with more context in hand.
    """

    llm = Ollama(model="llama3.1", temperature=0.3)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_text(transcript_text)
    
    # Summarize each chunk
    summaries = []
    for doc in docs:
        prompt = f"Please provide a concise summary of the following transcript excerpt:\n\n{doc}\n\nSummary:"
        summary = llm(prompt)
        summaries.append(summary)
    
    # Combine the chunk summaries into a final summary
    combined_summary = " ".join(summaries)
    
    # Optionally, summarize the combined summary
    final_prompt = f"Please provide an overall summary of the following text:\n\n{combined_summary}\n\nFinal Summary:"
    final_summary = llm(final_prompt)
    return final_summary


# Function to get conversational chain
def get_conversation_chain(vector_store,summary):
    """
    Function to create conversational chain using llama3.1 along with the prompt template and custom retriever.
    The prompt template consists of conversational memory, summary and context along with the user prompt.    
    """

    llm = Ollama(model="llama3.1", temperature=0.3)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, max_token_limit=2000)

    
    template = """You are a helpful assistant for a YouTube video chatbot.
                Use the following pieces of context and the summary to answer the human's question. If you don't know the answer, just say that you don't know; don't try to make up an answer.

                - Provide clear and concise answers.
                - Synthesize information from different parts of the video.
                - Include timestamps only when they are directly relevant and essential to your answer.
                - Avoid including too many timestamps or unnecessary details.
                - When mentioning timestamps, use the format [Timestamp: start - end], where 'start' and 'end' are in seconds and use two decimal places.

                Summary: {summary}

                Context: {context}

                Current conversation: {chat_history}
                Human: {question}
                AI Assistant:"""

    prompt = PromptTemplate(
        input_variables=["summary","context", "chat_history", "question"],
        template=template
    )

    # Create a partial prompt with the 'summary' filled in so that it need not be created everytime and is always inputed in the subsequent conversation chain
    partial_prompt = prompt.partial(summary=summary)

    retriever = CustomRetriever(vector_store=vector_store)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": partial_prompt}
    )
    return conversation_chain


def main():
    # Streamlit UI
    st.title("Youtube Video Chatbot")
    youtube_url = st.text_input("Enter Youtube Video URL:")

    if youtube_url:
        # Intial request and handling of video and starting up the conversation for this session
        try:
            video_id = extract_video_id(youtube_url)

            # get or create vector store 
            if 'vector_store' not in st.session_state:
                with st.spinner("Processing video transcript..."):  
                    transcript = get_transcript(video_id)
                    if transcript is None:
                        st.error("Sorry, we cannot help with this video as the transcript is not available.")
                        st.stop()
                    transcript_text = " ".join([seg['text'] for seg in transcript])
                    st.session_state.summary = create_video_summary(transcript_text)
                    st.session_state.vector_store = create_vector_store(transcript)

            # get or create conversation chain 
            if 'conversation' not in st.session_state:
                st.session_state.conversation = get_conversation_chain(st.session_state.vector_store,st.session_state.summary)

        except ValueError as e:
            st.error(f"Error: {str(e)}")
            st.stop()
        
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.stop()


        # Chat Interface for having the conversation
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about the video:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = st.session_state.conversation({'question': prompt})
                answer = response['answer']

                # Updated regex and replacement to replace timestamp format in the final output 
                timestamps = re.findall(r'\[Timestamp: (\d+\.\d{2}) - (\d+\.\d{2})\]', answer)
                for start, end in timestamps:
                    formatted_start = format_time(float(start))
                    formatted_end = format_time(float(end))
                    # Replace timestamps in the answer
                    original_timestamp = f"[Timestamp: {start} - {end}]"
                    formatted_timestamp = f"[Timestamp: {formatted_start} - {formatted_end}]"
                    answer = answer.replace(original_timestamp, formatted_timestamp)

                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        st.write("Please enter a YouTube video URL to start chatting")



if __name__ == "__main__":
    main()
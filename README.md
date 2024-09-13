# YouTube Video Chatbot

## Summary
YouTube Video Chatbot is a Streamlit-based application that allows users to interact with and ask questions about YouTube video content using natural language. The chatbot processes the video transcript, creates a summary, and uses advanced language models to provide relevant answers based on the video's content.

## Purpose
The main purpose of this application is to enhance the video-watching experience by allowing users to quickly extract information from YouTube videos without having to watch the entire content. It's particularly useful for educational videos, lectures, or any content where users might have specific questions about certain parts of the video.

## Features
- Extract and process YouTube video transcripts
- Generate video summaries using AI
- Create a vector store for efficient information retrieval
- Implement a conversational AI interface for user interactions
- Provide timestamp-aware responses to user queries
- Handle various YouTube URL formats

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/youtube-video-chatbot.git
   cd youtube-video-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have Ollama installed and the llama3.1 model available.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501).

3. Enter a YouTube video URL in the input field.

4. Once the video is processed, you can start asking questions about the video content in the chat interface.

## Sample Output

Here's an example of how the chat interface might look:

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
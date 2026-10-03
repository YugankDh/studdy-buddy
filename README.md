# Study Buddy: Chat with Your PDF Notes

A simple RAG (Retrieval-Augmented Generation) app built with Streamlit. Upload a PDF, then either **ask questions** or **summarize a topic**. Answers come only from your document and show the pages they came from.

## Features

- **Ask mode:** type a question and get an answer based only on your PDF. If the answer isn't in the document, the app says so.
- **Summarize mode:** type a topic and get a summary of just that part of the document.
- **Semantic search:** finds relevant sections by meaning, not exact keywords.
- **Source pages:** shows which pages the answer was taken from.
- **Fast reruns:** the PDF is embedded once and cached, so only short queries are embedded on each request.

## How It Works

1. The uploaded PDF is read and split into overlapping chunks (800 characters, 150 overlap).
2. Each chunk is converted into a vector with the `all-MiniLM-L6-v2` embedding model.
3. When you ask something, your query is embedded the same way and compared to all chunks using cosine similarity.
4. The top matching chunks are inserted into a prompt template as context.
5. Gemini generates the answer or summary using only that context.

```
PDF -> chunks -> embeddings -> similarity search -> top chunks -> prompt -> Gemini -> answer
```

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
- scikit-learn (cosine similarity)

## Run Locally

1. Clone the repo
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Add your API key. Create a `.env` file in the project folder:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
   ```

4. Run the app
   ```bash
   streamlit run main.py
   ```

## Limitations

- Works with text-based PDFs only. Scanned PDFs (images) need OCR, which isn't included.
- Each question is answered independently, so there is no chat history.
- The demo uses a shared API key with limited quota. Please don't upload private documents.

## Future Improvements

- Quiz mode that generates MCQs from a topic
- Summary style options (short, bullet points, exam-style)
- Multiple PDF support
- Chat history for follow-up questions

## Author

**Yugank**
[LinkedIn](add-your-link) | [GitHub](add-your-link)

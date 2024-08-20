# NLP Customer Feedback Pipeline

This project implements an Natural Language Processing (NLP) pipeline for analyzing customer feedback. It includes sentiment analysis, topic classification, keyword extraction, named entity recognition, aspect-based sentiment analysis, and text summarization.

## Features

- Sentiment analysis using DistilBERT
- Zero-shot topic classification
- Keyword extraction
- Named entity recognition
- Aspect-based sentiment analysis
- Text summarization
- Interactive Streamlit dashboard
- MongoDB integration for data storage

## Requirements

- Python 3.7+
- MongoDB

## Installation


1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with the following contents:
   ```
   MONGODB_URI=your_mongodb_uri
   MONGODB_DB=your_database_name
   MONGODB_COLLECTION=your_collection_name
   ```

## Usage

To run the Streamlit dashboard:

```
streamlit run app.py
```

Navigate to the URL provided by Streamlit (usually `http://localhost:8501`) to access the dashboard.

## Project Structure

- `app.py`: Main application file containing the NLP pipeline and Streamlit dashboard
- `requirements.txt`: List of Python package dependencies
- `README.md`: This file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

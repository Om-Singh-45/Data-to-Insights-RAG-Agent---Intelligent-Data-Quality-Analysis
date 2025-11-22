# Data-to-Insights RAG Agent: Intelligent Data Quality Analysis

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-000000?style=for-the-badge&logo=openai&logoColor=white)](https://openrouter.ai/)

Transform your raw data into actionable insights with this AI-powered data analysis platform. Upload your CSV or Excel files, and let our intelligent agent clean, analyze, and answer questions about your data using Retrieval-Augmented Generation (RAG) technology.

## 🌟 Key Features

### 🧹 Intelligent Data Cleaning
- **Smart Data Sanitization**: Automatically detects and handles special characters, null values, and zero-filled entries
- **Duplicate Removal**: Identifies and eliminates redundant records
- **Column Optimization**: Removes columns with excessive missing data (>50%)
- **Format Standardization**: Converts column names to consistent snake_case format
- **Date Processing**: Automatically identifies and converts date columns

### 📊 Automated Analytics Dashboard
- **Data Quality Heatmaps**: Visualize data completeness and integrity
- **Missing Value Patterns**: Identify patterns in missing data
- **Distribution Analysis**: Pie charts for categorical data distributions
- **Trend Analysis**: Line charts for temporal data trends
- **Outlier Detection**: Box plots to identify statistical anomalies
- **Frequency Distributions**: Histograms for numerical data analysis

### 🤖 AI-Powered Question Answering
- **Natural Language Queries**: Ask questions like "What are the top 5 products by revenue?"
- **RAG Technology**: Uses Retrieval-Augmented Generation for accurate, context-aware answers
- **Intelligent Visualization**: Automatically generates relevant charts based on your questions
- **Source Attribution**: See the data sources behind each answer

### 🔧 Flexible Embedding Options
- **Google Gemini Embeddings**: Default option using Google's powerful embedding models
- **Local HuggingFace Embeddings**: Offline option for privacy-conscious environments
- **Configurable Providers**: Easily switch between embedding providers via environment variables

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key (for embeddings)
- OpenRouter API key (for LLM queries) - Optional if using Gemini LLM

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd data-to-insights-rag-agent
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the application:**
```bash
streamlit run app.py
```

### Environment Variables

Create a `.env` file with the following variables:

```env
# Required for Gemini embeddings
GEMINI_API_KEY=your_gemini_api_key_here

# Required for OpenRouter LLM (can use Gemini LLM as alternative)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Choose embedding provider (gemini or hf)
EMBEDDING_PROVIDER=gemini
```

## 📖 How It Works

1. **Upload Data**: Upload your CSV or Excel file through the intuitive web interface
2. **Automatic Processing**: The system performs intelligent data cleaning and quality analysis
3. **Index Creation**: Your data is indexed using advanced embedding models for fast retrieval
4. **Ask Questions**: Use natural language to query your data and receive AI-generated insights
5. **Visualize Results**: Get automatically generated charts and visualizations tailored to your queries

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/) for responsive web interface
- **Data Processing**: [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)
- **Visualization**: [Plotly](https://plotly.com/) for interactive charts
- **AI/ML**: 
  - [Google Gemini](https://ai.google.dev/) for embeddings and LLM
  - [LlamaIndex](https://www.llamaindex.ai/) for RAG implementation
  - [HuggingFace](https://huggingface.co/) embeddings (optional)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) for efficient similarity search
- **Deployment**: Streamlit Cloud compatible

## 🎯 Use Cases

- **Business Intelligence**: Analyze sales data, customer behavior, and market trends
- **Data Quality Assessment**: Identify and resolve data integrity issues
- **Research Analysis**: Process and understand complex datasets
- **Educational Purposes**: Learn about data analysis through interactive exploration
- **Quick Insights**: Get immediate answers from your data without complex querying

## 📸 Screenshots

*Data Cleaning Overview*
![Data Cleaning]<img width="1838" height="812" alt="Screenshot 2025-11-22 211140" src="https://github.com/user-attachments/assets/eb96e3b0-39fc-4ee5-8a77-a563f209c044" />
<img width="1861" height="793" alt="Screenshot 2025-11-22 211201" src="https://github.com/user-attachments/assets/0c98ccea-ff7b-4cce-b1fc-d116b3433883" />



*Dashboard Overview*
![Dashboard]<img width="1811" height="854" alt="Screenshot 2025-11-22 211410" src="https://github.com/user-attachments/assets/7e04f369-01f2-4414-b234-106608cec7b6" />
<img width="1801" height="786" alt="Screenshot 2025-11-22 211435" src="https://github.com/user-attachments/assets/f216ae12-6cd6-4942-98ef-b9e3edbb1153" />

*Data Quality Analysis*
![Data Quality] <img width="1878" height="784" alt="Screenshot 2025-11-22 211452" src="https://github.com/user-attachments/assets/291f2273-144f-4712-bc15-394591e14bd6" />

*Question Answering*
![Q&A] <img width="1828" height="855" alt="Screenshot 2025-11-22 211816" src="https://github.com/user-attachments/assets/38ef9dac-1a0b-41c2-8c87-f6956e07301a" />

*Intelligent Visualization*
![Visualization] <img width="1829" height="875" alt="Screenshot 2025-11-22 211903" src="https://github.com/user-attachments/assets/320e77d2-383b-4cfc-a7ce-4147c681b7a4" />

## 🔒 Privacy & Security

- **Local Processing**: Data processing happens on your machine
- **API Key Management**: Secure environment variable configuration
- **No Data Storage**: Your data is not stored on external servers
- **Configurable Embeddings**: Choose between cloud (Gemini) and local (HuggingFace) embeddings

## 🤝 Contributing

We welcome contributions to improve the Data-to-Insights RAG Agent! Here's how you can help:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

Please ensure your code follows our style guidelines and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing libraries that make this project possible
- Special recognition to Google Gemini, LlamaIndex, and Streamlit teams
- Inspired by the need for accessible data analysis tools

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation for troubleshooting guides

---

*Transform your data into insights today with the Data-to-Insights RAG Agent!*

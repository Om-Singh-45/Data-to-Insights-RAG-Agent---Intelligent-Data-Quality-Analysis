# Data-to-Insights RAG Agent: Intelligent Data Quality Analysis


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

*Dashboard Overview*
![Dashboard](screenshots/dashboard.png)

*Data Quality Analysis*
![Data Quality](screenshots/data_quality.png)

*Question Answering*
![Q&A](screenshots/question_answering.png)

*Intelligent Visualization*
![Visualization](screenshots/visualization.png)

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
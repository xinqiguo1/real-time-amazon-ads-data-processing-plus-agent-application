# Real-Time Amazon Ads Data Processing & Agent Application

This project demonstrates a real-time data processing system for Amazon Ads API data, including ETL pipelines, performance analysis, and a multi-agent system for natural language interaction.

## Project Overview

This system processes real-time advertising data from Amazon Ads API to provide insights, performance monitoring, and optimization recommendations. The project includes:

1. **Real-time data processing** - ETL pipelines using PySpark and AWS Glue
2. **Performance analysis** - Automated algorithms to analyze ad performance metrics
3. **Multi-agent system** - Natural language interface for querying ad performance and getting recommendations

## Architecture Components

### Data Processing Pipeline

- **AWS Glue ETL** - Scheduled crawlers extract data from Amazon Ads API
- **Kinesis Data Firehose** - Real-time data streaming for immediate processing
- **S3 Storage** - Layered storage structure for raw and processed data
- **PySpark Processing** - Data transformation and analysis scripts

### Performance Analysis System

- **Lambda Functions** - Serverless functions for real-time analysis
- **Athena Queries** - SQL analytics on the advertising data
- **SNS Notifications** - Automated alerts for performance thresholds
- **Dynamic Bidding** - Algorithmic bid adjustments based on performance

### Multi-Agent System

- **LangChain Framework** - Orchestrates the agent system
- **Specialized Tools** - Custom tools for data retrieval and analysis
- **Natural Language Interface** - Allows non-technical users to query data
- **Recommendation Engine** - Provides optimization suggestions based on data analysis

## Key Files

- `amazon_ads_etl.py` - PySpark ETL script for data processing
- `ad_performance.py` - Lambda function for real-time performance analysis
- `amazon_ads_agent_system.py` - Multi-agent system for natural language interaction
- `requirements.txt` - Required Python dependencies

## Getting Started

### Prerequisites

- Python 3.8+
- AWS account with appropriate permissions
- Amazon Ads API access

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/real-time-amazon-ads-data-processing-plus-agent-application.git
cd real-time-amazon-ads-data-processing-plus-agent-application
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Configure AWS credentials:
```
aws configure
```

### Configuration

Update the following configuration parameters in the respective files:

- S3 bucket names
- Database names
- SNS topic ARNs
- API endpoints

## Usage

### Running the ETL Process

```python
spark-submit amazon_ads_etl.py
```

### Deploying Lambda Functions

Package the Lambda functions and deploy to AWS:

```bash
zip -r ad_performance.zip ad_performance.py
aws lambda update-function-code --function-name amazon-ads-performance --zip-file fileb://ad_performance.zip
```

### Using the Multi-Agent System

```python
python amazon_ads_agent_system.py
```

Example queries:
- "What is our overall advertising performance today?"
- "Which campaigns have the highest ACOS?"
- "Recommend bid adjustments for underperforming keywords"

## License

This project is for demonstration purposes only.

## Acknowledgments

- Amazon Ads API documentation
- AWS documentation for serverless architectures

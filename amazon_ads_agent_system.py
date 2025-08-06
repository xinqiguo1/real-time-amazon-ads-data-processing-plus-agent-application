import os
import json
import boto3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS clients
athena_client = boto3.client('athena')
s3_client = boto3.client('s3')
sns_client = boto3.client('sns')
lambda_client = boto3.client('lambda')

# Configuration
ATHENA_OUTPUT_LOCATION = "s3://default-athena-result/Unsaved/agent_output"
ATHENA_DATABASE = "amazon-stream"
PERFORMANCE_HISTORY_BUCKET = "amazon-ads-performance-history"
PERFORMANCE_HISTORY_KEY = "performance_history.json"
BIDDING_STRATEGY_BUCKET = "amazon-ads-bidding-strategies"
BIDDING_STRATEGY_KEY = "bidding_rules.json"
SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:412381752211:Ad_Performance_Alert"

# Agent-specific constants
AGENT_SYSTEM_PROMPT = """You are an Amazon Advertising specialist AI assistant. Your role is to help users understand their 
advertising performance, provide insights, and recommend optimization strategies. You have access to real-time and historical 
advertising data from Amazon Ads API. Always be helpful, clear, and provide actionable recommendations based on data."""

# Tool definitions for the agents
class QueryAthenaDataTool(BaseTool):
    name = "query_athena_data"
    description = "Execute an Athena SQL query to retrieve Amazon Ads performance data"
    
    def _run(self, query: str) -> List[Dict]:
        """Execute the Athena query and return results"""
        try:
            # Start the query execution
            response = athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={'Database': ATHENA_DATABASE},
                ResultConfiguration={'OutputLocation': ATHENA_OUTPUT_LOCATION}
            )
            
            query_execution_id = response["QueryExecutionId"]
            logger.info(f"Started Athena query with execution ID: {query_execution_id}")
            
            # Wait for query to complete
            import time
            while True:
                query_status = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
                status = query_status["QueryExecution"]["Status"]["State"]
                
                if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                    break
                time.sleep(1)  # Check every second
            
            if status != "SUCCEEDED":
                error_message = query_status["QueryExecution"]["Status"].get("StateChangeReason", "Unknown error")
                logger.error(f"Query failed with status {status}: {error_message}")
                return {"error": f"Query failed with status {status}: {error_message}"}
            
            # Fetch the query results
            results = athena_client.get_query_results(QueryExecutionId=query_execution_id)
            
            # Process the results
            headers = [col["VarCharValue"] for col in results["ResultSet"]["Rows"][0]["Data"]]
            
            processed_results = []
            for i in range(1, len(results["ResultSet"]["Rows"])):
                row = results["ResultSet"]["Rows"][i]["Data"]
                row_data = {headers[j]: row[j].get("VarCharValue", None) for j in range(len(headers))}
                processed_results.append(row_data)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error executing Athena query: {str(e)}")
            return {"error": str(e)}
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class GetPerformanceSummaryTool(BaseTool):
    name = "get_performance_summary"
    description = "Get a summary of current Amazon Ads performance metrics"
    
    def _run(self, time_window: str = "today") -> Dict:
        """Get performance summary for the specified time window"""
        # Determine date filter based on time window
        if time_window == "today":
            date_filter = "DATE(from_iso8601_timestamp(time_window_start)) = current_date"
        elif time_window == "yesterday":
            date_filter = "DATE(from_iso8601_timestamp(time_window_start)) = current_date - interval '1' day"
        elif time_window == "last7days":
            date_filter = "DATE(from_iso8601_timestamp(time_window_start)) >= current_date - interval '7' day"
        else:
            date_filter = "DATE(from_iso8601_timestamp(time_window_start)) = current_date"
        
        # Build the query
        query = f"""
        with sp_conv as 
        (
            SELECT 
                sum(attributed_conversions_1d) total_order, 
                sum(attributed_sales_1d) total_sales,
                sum(attributed_units_ordered_1d) total_units
            FROM "{ATHENA_DATABASE}"."sp_conversion_na_firehose_s3"
            WHERE {date_filter}
        ),
        sp_traff as
        (
            SELECT 
                sum(impressions) total_impression, 
                sum(cost) total_cost, 
                sum(clicks) total_click
            FROM "{ATHENA_DATABASE}"."sp_traffic_firehose_s3"
            WHERE {date_filter}
        )

        SELECT   
            t.total_impression,
            t.total_click,
            round(t.total_cost,2) AS spend,
            c.total_order,
            round(c.total_sales,2) AS sales,
            c.total_units,
            CASE WHEN c.total_sales > 0 THEN round((t.total_cost*100.0/c.total_sales),2) ELSE 0 END AS acos,
            CASE WHEN t.total_cost > 0 THEN round((c.total_sales/t.total_cost),2) ELSE 0 END AS roas,
            CASE WHEN t.total_click > 0 THEN round((c.total_order*100.0/t.total_click),2) ELSE 0 END AS conversion,
            CASE WHEN t.total_impression > 0 THEN round((t.total_click*100.0/t.total_impression),2) ELSE 0 END AS ctr
        FROM sp_traff t, sp_conv c
        """
        
        try:
            query_tool = QueryAthenaDataTool()
            result = query_tool._run(query)
            
            if isinstance(result, list) and len(result) > 0:
                return result[0]
            else:
                return {"error": "No performance data available"}
        
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {"error": str(e)}
    
    def _arun(self, time_window: str = "today"):
        raise NotImplementedError("This tool does not support async")

class GetCampaignPerformanceTool(BaseTool):
    name = "get_campaign_performance"
    description = "Get performance metrics for specific campaigns or all campaigns"
    
    def _run(self, campaign_id: str = None, time_window: str = "today") -> List[Dict]:
        """Get campaign performance for the specified campaign(s) and time window"""
        # Determine date filter based on time window
        if time_window == "today":
            date_filter = "DATE(from_iso8601_timestamp(time_window_start)) = current_date"
        elif time_window == "yesterday":
            date_filter = "DATE(from_iso8601_timestamp(time_window_start)) = current_date - interval '1' day"
        elif time_window == "last7days":
            date_filter = "DATE(from_iso8601_timestamp(time_window_start)) >= current_date - interval '7' day"
        else:
            date_filter = "DATE(from_iso8601_timestamp(time_window_start)) = current_date"
        
        # Add campaign filter if specified
        campaign_filter = f"AND t.campaign_id = '{campaign_id}'" if campaign_id else ""
        
        # Build the query
        query = f"""
        with sp_conv as 
        (
            SELECT 
                campaign_id,
                sum(attributed_conversions_1d) total_order, 
                sum(attributed_sales_1d) total_sales,
                sum(attributed_units_ordered_1d) total_units
            FROM "{ATHENA_DATABASE}"."sp_conversion_na_firehose_s3"
            WHERE {date_filter}
            GROUP BY campaign_id
        ),
        sp_traff as
        (
            SELECT 
                campaign_id,
                sum(impressions) total_impression, 
                sum(cost) total_cost, 
                sum(clicks) total_click
            FROM "{ATHENA_DATABASE}"."sp_traffic_firehose_s3"
            WHERE {date_filter}
            GROUP BY campaign_id
        )

        SELECT   
            t.campaign_id,
            t.total_impression,
            t.total_click,
            round(t.total_cost,2) AS spend,
            c.total_order,
            round(c.total_sales,2) AS sales,
            c.total_units,
            CASE WHEN c.total_sales > 0 THEN round((t.total_cost*100.0/c.total_sales),2) ELSE 0 END AS acos,
            CASE WHEN t.total_cost > 0 THEN round((c.total_sales/t.total_cost),2) ELSE 0 END AS roas,
            CASE WHEN t.total_click > 0 THEN round((c.total_order*100.0/t.total_click),2) ELSE 0 END AS conversion,
            CASE WHEN t.total_impression > 0 THEN round((t.total_click*100.0/t.total_impression),2) ELSE 0 END AS ctr
        FROM sp_traff t
        LEFT JOIN sp_conv c ON t.campaign_id = c.campaign_id
        WHERE 1=1 {campaign_filter}
        ORDER BY spend DESC
        """
        
        try:
            query_tool = QueryAthenaDataTool()
            result = query_tool._run(query)
            return result
        
        except Exception as e:
            logger.error(f"Error getting campaign performance: {str(e)}")
            return {"error": str(e)}
    
    def _arun(self, campaign_id: str = None, time_window: str = "today"):
        raise NotImplementedError("This tool does not support async")

class GetKeywordPerformanceTool(BaseTool):
    name = "get_keyword_performance"
    description = "Get performance metrics for keywords, optionally filtered by campaign"
    
    def _run(self, campaign_id: str = None, keyword: str = None, time_window: str = "today", limit: int = 10) -> List[Dict]:
        """Get keyword performance data"""
        # Determine date filter based on time window
        if time_window == "today":
            date_filter = "DATE(from_iso8601_timestamp(time_window_start)) = current_date"
        elif time_window == "yesterday":
            date_filter = "DATE(from_iso8601_timestamp(time_window_start)) = current_date - interval '1' day"
        elif time_window == "last7days":
            date_filter = "DATE(from_iso8601_timestamp(time_window_start)) >= current_date - interval '7' day"
        else:
            date_filter = "DATE(from_iso8601_timestamp(time_window_start)) = current_date"
        
        # Add filters if specified
        campaign_filter = f"AND t.campaign_id = '{campaign_id}'" if campaign_id else ""
        keyword_filter = f"AND t.keyword_text LIKE '%{keyword}%'" if keyword else ""
        
        # Build the query
        query = f"""
        SELECT 
            t.campaign_id,
            t.keyword_id,
            t.keyword_text,
            t.match_type,
            sum(t.impressions) as impressions,
            sum(t.clicks) as clicks,
            sum(t.cost) as cost,
            sum(c.attributed_conversions_1d) as conversions,
            sum(c.attributed_sales_1d) as sales,
            CASE WHEN sum(c.attributed_sales_1d) > 0 
                THEN round((sum(t.cost)*100.0/sum(c.attributed_sales_1d)),2) 
                ELSE 0 
            END AS acos,
            CASE WHEN sum(t.cost) > 0 
                THEN round((sum(c.attributed_sales_1d)/sum(t.cost)),2) 
                ELSE 0 
            END AS roas,
            CASE WHEN sum(t.clicks) > 0 
                THEN round((sum(c.attributed_conversions_1d)*100.0/sum(t.clicks)),2) 
                ELSE 0 
            END AS conversion_rate,
            CASE WHEN sum(t.impressions) > 0 
                THEN round((sum(t.clicks)*100.0/sum(t.impressions)),2) 
                ELSE 0 
            END AS ctr
        FROM "{ATHENA_DATABASE}"."sp_traffic_firehose_s3" t
        LEFT JOIN "{ATHENA_DATABASE}"."sp_conversion_na_firehose_s3" c 
            ON t.campaign_id = c.campaign_id 
            AND t.ad_group_id = c.ad_group_id
            AND t.time_window_start = c.time_window_start
        WHERE {date_filter}
            AND t.keyword_text IS NOT NULL
            {campaign_filter}
            {keyword_filter}
        GROUP BY t.campaign_id, t.keyword_id, t.keyword_text, t.match_type
        ORDER BY cost DESC
        LIMIT {limit}
        """
        
        try:
            query_tool = QueryAthenaDataTool()
            result = query_tool._run(query)
            return result
        
        except Exception as e:
            logger.error(f"Error getting keyword performance: {str(e)}")
            return {"error": str(e)}
    
    def _arun(self, campaign_id: str = None, keyword: str = None, time_window: str = "today", limit: int = 10):
        raise NotImplementedError("This tool does not support async")

class GetHistoricalPerformanceTool(BaseTool):
    name = "get_historical_performance"
    description = "Get historical performance data from S3"
    
    def _run(self) -> Dict:
        """Retrieve historical performance data from S3"""
        try:
            response = s3_client.get_object(
                Bucket=PERFORMANCE_HISTORY_BUCKET,
                Key=PERFORMANCE_HISTORY_KEY
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            logger.error(f"Error retrieving historical data: {str(e)}")
            return {"error": f"Error retrieving historical data: {str(e)}"}
    
    def _arun(self):
        raise NotImplementedError("This tool does not support async")

class GetBiddingRecommendationsTool(BaseTool):
    name = "get_bidding_recommendations"
    description = "Get bidding recommendations based on campaign performance"
    
    def _run(self, campaign_id: str = None, time_window: str = "last7days") -> List[Dict]:
        """Get bidding recommendations for campaigns"""
        # First, get campaign performance data
        campaign_tool = GetCampaignPerformanceTool()
        campaigns = campaign_tool._run(campaign_id=campaign_id, time_window=time_window)
        
        if isinstance(campaigns, dict) and 'error' in campaigns:
            return campaigns
        
        # Get bidding rules
        try:
            response = s3_client.get_object(
                Bucket=BIDDING_STRATEGY_BUCKET,
                Key=BIDDING_STRATEGY_KEY
            )
            rules = json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            # Use default rules if can't retrieve from S3
            rules = {
                "default_rules": [
                    {
                        "condition": "roas < 1.0",
                        "action": "decrease_bid",
                        "percentage": 20,
                        "description": "ROAS below 1 is losing money, decrease bids by 20%"
                    },
                    {
                        "condition": "roas >= 4.0",
                        "action": "increase_bid",
                        "percentage": 10,
                        "description": "ROAS above 4 is performing well, increase bids by 10% to gain more volume"
                    },
                    {
                        "condition": "acos > 40.0",
                        "action": "decrease_bid",
                        "percentage": 15,
                        "description": "ACOS above 40% is concerning, decrease bids by 15%"
                    },
                    {
                        "condition": "conversion > 10.0 and roas >= 3.0",
                        "action": "increase_bid",
                        "percentage": 15,
                        "description": "High conversion rate with good ROAS, increase bids by 15% to gain more volume"
                    }
                ]
            }
        
        # Evaluate bidding rules for each campaign
        recommendations = []
        
        for campaign in campaigns:
            # Skip if missing necessary data
            if not all(key in campaign for key in ['campaign_id', 'acos', 'roas', 'conversion', 'ctr']):
                continue
                
            # Convert string values to float for comparison
            acos = float(campaign['acos']) if campaign['acos'] else 0
            roas = float(campaign['roas']) if campaign['roas'] else 0
            conversion = float(campaign['conversion']) if campaign['conversion'] else 0
            ctr = float(campaign['ctr']) if campaign['ctr'] else 0
            
            for rule in rules['default_rules']:
                condition = rule['condition']
                
                # Evaluate the condition
                try:
                    # Replace variables with actual values
                    eval_condition = condition.replace('acos', str(acos))
                    eval_condition = eval_condition.replace('roas', str(roas))
                    eval_condition = eval_condition.replace('conversion', str(conversion))
                    eval_condition = eval_condition.replace('ctr', str(ctr))
                    
                    if eval(eval_condition):
                        recommendations.append({
                            'campaign_id': campaign['campaign_id'],
                            'current_metrics': {
                                'acos': acos,
                                'roas': roas,
                                'conversion': conversion,
                                'ctr': ctr
                            },
                            'action': rule['action'],
                            'percentage': rule['percentage'],
                            'description': rule['description'],
                            'reason': f"Rule triggered: {rule['condition']}"
                        })
                except Exception as e:
                    logger.error(f"Error evaluating rule condition '{condition}': {str(e)}")
        
        return recommendations
    
    def _arun(self, campaign_id: str = None, time_window: str = "last7days"):
        raise NotImplementedError("This tool does not support async")

class SendAlertTool(BaseTool):
    name = "send_alert"
    description = "Send an alert via SNS about advertising performance issues"
    
    def _run(self, subject: str, message: str, importance: str = "normal") -> Dict:
        """Send an SNS alert"""
        try:
            # Add emoji based on importance
            if importance.lower() == "critical":
                message = f"🚨 CRITICAL ALERT 🚨\n\n{message}"
                subject = f"CRITICAL: {subject}"
            elif importance.lower() == "warning":
                message = f"⚠️ WARNING ⚠️\n\n{message}"
                subject = f"WARNING: {subject}"
            
            # Send SNS Alert
            response = sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=message,
                Subject=subject
            )
            
            return {"status": "success", "message_id": response['MessageId']}
            
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
            return {"error": str(e)}
    
    def _arun(self, subject: str, message: str, importance: str = "normal"):
        raise NotImplementedError("This tool does not support async")

# Custom prompt template for the agent
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        intermediate_steps = kwargs.pop("intermediate_steps")
        
        # Format the list of tools available
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # Format the list of previous steps
        history = ""
        for action, observation in intermediate_steps:
            history += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
        
        # Set the agent_scratchpad variable to the formatted history
        kwargs["agent_scratchpad"] = history
        kwargs["tools"] = tools_str
        
        # Create a tools lookup dictionary
        tools_lookup = {tool.name: tool for tool in self.tools}
        kwargs["tools_lookup"] = tools_lookup
        
        return self.template.format(**kwargs)

# LLM parser for the agent
def parse_output(llm_output: str) -> Union[AgentAction, AgentFinish]:
    # Check if the output indicates the agent should finish
    if "Final Answer:" in llm_output:
        return AgentFinish(
            return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
            log=llm_output,
        )
    
    # Parse the action and input
    regex = r"Action: (.*?)[\n]*Action Input: (.*)"
    match = re.search(regex, llm_output, re.DOTALL)
    
    if not match:
        # If no match is found, return a default finish response
        return AgentFinish(
            return_values={"output": "I couldn't determine what to do next. Please provide more information."},
            log=llm_output,
        )
    
    action = match.group(1).strip()
    action_input = match.group(2).strip()
    
    # Return the action and input
    return AgentAction(tool=action, tool_input=action_input, log=llm_output)

# Create the agent prompt
agent_prompt = CustomPromptTemplate(
    template="""
{system_prompt}

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""",
    tools=[
        QueryAthenaDataTool(),
        GetPerformanceSummaryTool(),
        GetCampaignPerformanceTool(),
        GetKeywordPerformanceTool(),
        GetHistoricalPerformanceTool(),
        GetBiddingRecommendationsTool(),
        SendAlertTool()
    ],
    input_variables=["input", "system_prompt", "agent_scratchpad"],
)

# Initialize the LLM
def get_llm():
    """Initialize and return the LLM"""
    try:
        # Try to use Bedrock Claude model
        llm = Bedrock(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-east-1",
            model_kwargs={"temperature": 0, "max_tokens": 4096}
        )
        return llm
    except Exception as e:
        logger.warning(f"Failed to initialize Bedrock LLM: {str(e)}. Falling back to OpenAI.")
        # Fall back to OpenAI
        try:
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0,
                max_tokens=4096
            )
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {str(e)}")
            raise Exception("Failed to initialize any LLM")

# Create the agent
def create_agent():
    """Create and return the agent executor"""
    llm = get_llm()
    
    # Create the tools
    tools = [
        QueryAthenaDataTool(),
        GetPerformanceSummaryTool(),
        GetCampaignPerformanceTool(),
        GetKeywordPerformanceTool(),
        GetHistoricalPerformanceTool(),
        GetBiddingRecommendationsTool(),
        SendAlertTool()
    ]
    
    # Create the LLM chain
    llm_chain = LLMChain(
        llm=llm,
        prompt=agent_prompt
    )
    
    # Create the agent
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=parse_output,
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools]
    )
    
    # Create the agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        handle_parsing_errors=True
    )
    
    return agent_executor

# Main handler for Lambda function
def lambda_handler(event, context):
    """
    Main Lambda handler function for the multi-agent system
    """
    try:
        # Extract the user query from the event
        user_query = event.get('query', '')
        
        if not user_query:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No query provided'})
            }
        
        # Create the agent
        agent_executor = create_agent()
        
        # Run the agent
        response = agent_executor.run(
            input=user_query,
            system_prompt=AGENT_SYSTEM_PROMPT
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'response': response
            })
        }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

# For local testing
if __name__ == "__main__":
    # Create the agent
    agent_executor = create_agent()
    
    # Test query
    test_query = "What is our overall advertising performance today? Are there any campaigns that need attention?"
    
    # Run the agent
    response = agent_executor.run(
        input=test_query,
        system_prompt=AGENT_SYSTEM_PROMPT
    )
    
    print(f"Response: {response}") 
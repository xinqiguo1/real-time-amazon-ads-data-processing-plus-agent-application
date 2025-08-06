import boto3
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS clients
athena_client = boto3.client('athena')
sns_client = boto3.client('sns')
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')

# Configuration
SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:412381752211:Ad_Spending_Alert"
ATHENA_OUTPUT_LOCATION = "s3://default-athena-result/Unsaved/lambda_output"
ATHENA_DATABASE = "amazon-stream"
PERFORMANCE_HISTORY_BUCKET = "amazon-ads-performance-history"
PERFORMANCE_HISTORY_KEY = "performance_history.json"
BIDDING_STRATEGY_BUCKET = "amazon-ads-bidding-strategies"
BIDDING_STRATEGY_KEY = "bidding_rules.json"

# Thresholds for alerts
THRESHOLDS = {
    "ROAS": {
        "critical_low": 1.0,  # ROAS below 1 is losing money
        "warning_low": 2.0,   # ROAS below 2 is concerning
        "target": 3.0,        # Target ROAS
        "good": 4.0           # Good ROAS
    },
    "ACOS": {
        "critical_high": 50.0,  # ACOS above 50% is concerning
        "warning_high": 35.0,   # ACOS above 35% needs attention
        "target": 25.0,         # Target ACOS
        "good": 20.0            # Good ACOS
    },
    "CTR": {
        "critical_low": 0.1,    # CTR below 0.1% is very poor
        "warning_low": 0.25,    # CTR below 0.25% is concerning
        "target": 0.5,          # Target CTR
        "good": 0.75            # Good CTR
    },
    "Conversion": {
        "critical_low": 1.0,    # Conversion below 1% is concerning
        "warning_low": 2.0,     # Conversion below 2% needs attention
        "target": 3.0,          # Target conversion rate
        "good": 5.0             # Good conversion rate
    },
    "Daily_Spend": {
        "warning_high": 500.0,  # Alert if daily spend exceeds $500
        "critical_high": 1000.0 # Critical alert if daily spend exceeds $1000
    }
}

def execute_athena_query(query, database):
    """
    Execute an Athena query and return the results
    """
    try:
        # Start the query execution
        response = athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': database},
            ResultConfiguration={'OutputLocation': ATHENA_OUTPUT_LOCATION}
        )
        
        query_execution_id = response["QueryExecutionId"]
        logger.info(f"Started Athena query with execution ID: {query_execution_id}")
        
        # Wait for query to complete
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

def get_performance_data(time_window="today"):
    """
    Get performance data for the specified time window
    """
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
            campaign_id,
            ad_group_id,
            sum(attributed_conversions_1d) total_order, 
            sum(attributed_sales_1d) total_sales,
            sum(attributed_units_ordered_1d) total_units
        FROM "{ATHENA_DATABASE}"."sp_conversion_na_firehose_s3"
        WHERE {date_filter}
        GROUP BY campaign_id, ad_group_id
    ),
    sp_traff as
    (
        SELECT 
            campaign_id,
            ad_group_id,
            sum(impressions) total_impression, 
            sum(cost) total_cost, 
            sum(clicks) total_click
        FROM "{ATHENA_DATABASE}"."sp_traffic_firehose_s3"
        WHERE {date_filter}
        GROUP BY campaign_id, ad_group_id
    )

    SELECT   
        t.campaign_id,
        t.ad_group_id,
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
    LEFT JOIN sp_conv c ON t.campaign_id = c.campaign_id AND t.ad_group_id = c.ad_group_id
    """
    
    return execute_athena_query(query, ATHENA_DATABASE)

def get_historical_performance():
    """
    Retrieve historical performance data from S3
    """
    try:
        response = s3_client.get_object(
            Bucket=PERFORMANCE_HISTORY_BUCKET,
            Key=PERFORMANCE_HISTORY_KEY
        )
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"No historical data found at s3://{PERFORMANCE_HISTORY_BUCKET}/{PERFORMANCE_HISTORY_KEY}")
            return {}
        else:
            logger.error(f"Error retrieving historical data: {str(e)}")
            return {}

def save_historical_performance(performance_data):
    """
    Save performance data to S3 for historical tracking
    """
    try:
        # Add timestamp
        performance_data['timestamp'] = datetime.now().isoformat()
        
        # Get existing history
        history = get_historical_performance()
        
        # If no history exists, initialize it
        if not history:
            history = {'performance_history': []}
        
        # Add new data
        history['performance_history'].append(performance_data)
        
        # Keep only last 30 days of data
        if len(history['performance_history']) > 30:
            history['performance_history'] = history['performance_history'][-30:]
        
        # Save to S3
        s3_client.put_object(
            Bucket=PERFORMANCE_HISTORY_BUCKET,
            Key=PERFORMANCE_HISTORY_KEY,
            Body=json.dumps(history),
            ContentType='application/json'
        )
        logger.info(f"Successfully saved performance history to S3")
        
    except Exception as e:
        logger.error(f"Error saving historical performance: {str(e)}")

def get_bidding_rules():
    """
    Retrieve bidding rules from S3
    """
    try:
        response = s3_client.get_object(
            Bucket=BIDDING_STRATEGY_BUCKET,
            Key=BIDDING_STRATEGY_KEY
        )
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"No bidding rules found at s3://{BIDDING_STRATEGY_BUCKET}/{BIDDING_STRATEGY_KEY}")
            # Return default rules
            return {
                "default_rules": [
                    {
                        "condition": "roas < 1.0",
                        "action": "decrease_bid",
                        "percentage": 20
                    },
                    {
                        "condition": "roas >= 4.0",
                        "action": "increase_bid",
                        "percentage": 10
                    },
                    {
                        "condition": "acos > 40.0",
                        "action": "decrease_bid",
                        "percentage": 15
                    },
                    {
                        "condition": "conversion > 10.0 and roas >= 3.0",
                        "action": "increase_bid",
                        "percentage": 15
                    }
                ]
            }
        else:
            logger.error(f"Error retrieving bidding rules: {str(e)}")
            return {}

def evaluate_bidding_rules(performance_data, rules):
    """
    Evaluate bidding rules against performance data and return bid adjustments
    """
    bid_adjustments = []
    
    # Default rules if none provided
    if not rules or 'default_rules' not in rules:
        logger.warning("No valid bidding rules found, using default rules")
        return []
    
    for campaign in performance_data:
        # Skip if missing necessary data
        if not all(key in campaign for key in ['campaign_id', 'ad_group_id', 'acos', 'roas', 'conversion']):
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
                condition = condition.replace('acos', str(acos))
                condition = condition.replace('roas', str(roas))
                condition = condition.replace('conversion', str(conversion))
                condition = condition.replace('ctr', str(ctr))
                
                if eval(condition):
                    bid_adjustments.append({
                        'campaign_id': campaign['campaign_id'],
                        'ad_group_id': campaign['ad_group_id'],
                        'action': rule['action'],
                        'percentage': rule['percentage'],
                        'reason': f"Rule triggered: {rule['condition']}"
                    })
            except Exception as e:
                logger.error(f"Error evaluating rule condition '{condition}': {str(e)}")
    
    return bid_adjustments

def apply_bid_adjustments(bid_adjustments):
    """
    Apply bid adjustments via Lambda function
    """
    if not bid_adjustments:
        logger.info("No bid adjustments to apply")
        return
    
    try:
        # Invoke a Lambda function to apply the bid adjustments
        response = lambda_client.invoke(
            FunctionName='amazon-ads-bid-adjuster',
            InvocationType='Event',  # Asynchronous invocation
            Payload=json.dumps({'bid_adjustments': bid_adjustments})
        )
        logger.info(f"Successfully invoked bid adjustment Lambda with {len(bid_adjustments)} adjustments")
    except Exception as e:
        logger.error(f"Error applying bid adjustments: {str(e)}")

def generate_performance_summary(performance_data):
    """
    Generate a summary of performance data
    """
    if not performance_data or isinstance(performance_data, dict) and 'error' in performance_data:
        return {
            'error': 'No valid performance data available'
        }
    
    # Convert to pandas DataFrame for easier aggregation
    df = pd.DataFrame(performance_data)
    
    # Convert numeric columns
    numeric_cols = ['total_impression', 'total_click', 'spend', 'total_order', 
                   'sales', 'total_units', 'acos', 'roas', 'conversion', 'ctr']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate overall metrics
    summary = {
        'total_campaigns': df['campaign_id'].nunique(),
        'total_ad_groups': df['ad_group_id'].nunique(),
        'total_impressions': df['total_impression'].sum(),
        'total_clicks': df['total_click'].sum(),
        'total_spend': round(df['spend'].sum(), 2),
        'total_orders': df['total_order'].sum(),
        'total_sales': round(df['sales'].sum(), 2),
        'total_units': df['total_units'].sum() if 'total_units' in df.columns else 0,
        'overall_acos': round((df['spend'].sum() / df['sales'].sum()) * 100, 2) if df['sales'].sum() > 0 else 0,
        'overall_roas': round(df['sales'].sum() / df['spend'].sum(), 2) if df['spend'].sum() > 0 else 0,
        'overall_conversion': round((df['total_order'].sum() / df['total_click'].sum()) * 100, 2) if df['total_click'].sum() > 0 else 0,
        'overall_ctr': round((df['total_click'].sum() / df['total_impression'].sum()) * 100, 2) if df['total_impression'].sum() > 0 else 0
    }
    
    # Add performance distribution
    if len(df) > 0:
        summary['roas_distribution'] = {
            'low_performing': len(df[df['roas'] < THRESHOLDS['ROAS']['warning_low']]),
            'average_performing': len(df[(df['roas'] >= THRESHOLDS['ROAS']['warning_low']) & (df['roas'] < THRESHOLDS['ROAS']['good'])]),
            'high_performing': len(df[df['roas'] >= THRESHOLDS['ROAS']['good']])
        }
        
        summary['acos_distribution'] = {
            'high_acos': len(df[df['acos'] > THRESHOLDS['ACOS']['warning_high']]),
            'target_acos': len(df[(df['acos'] <= THRESHOLDS['ACOS']['warning_high']) & (df['acos'] > THRESHOLDS['ACOS']['good'])]),
            'low_acos': len(df[df['acos'] <= THRESHOLDS['ACOS']['good']])
        }
    
    return summary

def check_alerts(performance_data, summary):
    """
    Check for alert conditions and generate alert messages
    """
    alerts = []
    
    # Check overall performance alerts
    if summary['overall_roas'] < THRESHOLDS['ROAS']['critical_low']:
        alerts.append({
            'level': 'CRITICAL',
            'type': 'ROAS',
            'message': f"Overall ROAS is critically low at {summary['overall_roas']} (threshold: {THRESHOLDS['ROAS']['critical_low']})"
        })
    elif summary['overall_roas'] < THRESHOLDS['ROAS']['warning_low']:
        alerts.append({
            'level': 'WARNING',
            'type': 'ROAS',
            'message': f"Overall ROAS is below target at {summary['overall_roas']} (threshold: {THRESHOLDS['ROAS']['warning_low']})"
        })
    
    if summary['overall_acos'] > THRESHOLDS['ACOS']['critical_high']:
        alerts.append({
            'level': 'CRITICAL',
            'type': 'ACOS',
            'message': f"Overall ACOS is critically high at {summary['overall_acos']}% (threshold: {THRESHOLDS['ACOS']['critical_high']}%)"
        })
    elif summary['overall_acos'] > THRESHOLDS['ACOS']['warning_high']:
        alerts.append({
            'level': 'WARNING',
            'type': 'ACOS',
            'message': f"Overall ACOS is above target at {summary['overall_acos']}% (threshold: {THRESHOLDS['ACOS']['warning_high']}%)"
        })
    
    if summary['total_spend'] > THRESHOLDS['Daily_Spend']['critical_high']:
        alerts.append({
            'level': 'CRITICAL',
            'type': 'SPEND',
            'message': f"Daily spend is critically high at ${summary['total_spend']} (threshold: ${THRESHOLDS['Daily_Spend']['critical_high']})"
        })
    elif summary['total_spend'] > THRESHOLDS['Daily_Spend']['warning_high']:
        alerts.append({
            'level': 'WARNING',
            'type': 'SPEND',
            'message': f"Daily spend is high at ${summary['total_spend']} (threshold: ${THRESHOLDS['Daily_Spend']['warning_high']})"
        })
    
    # Check for individual campaign alerts
    if isinstance(performance_data, list):
        for campaign in performance_data:
            # Convert string values to float
            acos = float(campaign['acos']) if campaign['acos'] else 0
            roas = float(campaign['roas']) if campaign['roas'] else 0
            spend = float(campaign['spend']) if campaign['spend'] else 0
            
            # High-spending campaigns with poor performance
            if spend > 100 and roas < THRESHOLDS['ROAS']['warning_low']:
                alerts.append({
                    'level': 'WARNING',
                    'type': 'CAMPAIGN',
                    'campaign_id': campaign['campaign_id'],
                    'message': f"Campaign {campaign['campaign_id']} has high spend (${spend}) with low ROAS ({roas})"
                })
            
            # Extremely poor performing campaigns
            if spend > 50 and roas < THRESHOLDS['ROAS']['critical_low']:
                alerts.append({
                    'level': 'CRITICAL',
                    'type': 'CAMPAIGN',
                    'campaign_id': campaign['campaign_id'],
                    'message': f"Campaign {campaign['campaign_id']} has very poor performance: ROAS {roas}, ACOS {acos}%"
                })
    
    return alerts

def send_alerts(alerts):
    """
    Send alerts via SNS
    """
    if not alerts:
        logger.info("No alerts to send")
        return
    
    # Group alerts by level
    critical_alerts = [a for a in alerts if a['level'] == 'CRITICAL']
    warning_alerts = [a for a in alerts if a['level'] == 'WARNING']
    
    # Prepare alert messages
    if critical_alerts:
        alert_message = "🚨 CRITICAL ALERTS 🚨\n\n"
        for alert in critical_alerts:
            alert_message += f"- {alert['message']}\n"
        
        alert_message += "\nImmediate action recommended!"
        
        # Send SNS Alert
        try:
            response = sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=alert_message,
                Subject="CRITICAL Amazon Ads Performance Alert"
            )
            logger.info(f"Sent {len(critical_alerts)} critical alerts via SNS")
        except Exception as e:
            logger.error(f"Error sending critical alerts: {str(e)}")
    
    if warning_alerts:
        alert_message = "⚠️ WARNING ALERTS ⚠️\n\n"
        for alert in warning_alerts:
            alert_message += f"- {alert['message']}\n"
        
        alert_message += "\nPlease review and take appropriate action."
        
        # Send SNS Alert
        try:
            response = sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=alert_message,
                Subject="Amazon Ads Performance Warning"
            )
            logger.info(f"Sent {len(warning_alerts)} warning alerts via SNS")
        except Exception as e:
            logger.error(f"Error sending warning alerts: {str(e)}")

def lambda_handler(event, context):
    """
    Main Lambda handler function
    """
    logger.info("Starting Amazon Ads performance analysis")
    
    # Get performance data
    time_window = event.get('time_window', 'today')
    performance_data = get_performance_data(time_window)
    
    if isinstance(performance_data, dict) and 'error' in performance_data:
        logger.error(f"Error retrieving performance data: {performance_data['error']}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f"Error retrieving performance data: {performance_data['error']}"
            })
        }
    
    # Generate performance summary
    summary = generate_performance_summary(performance_data)
    
    # Save historical data
    save_historical_performance(summary)
    
    # Check for alerts
    alerts = check_alerts(performance_data, summary)
    
    # Send alerts if needed
    send_alerts(alerts)
    
    # Get bidding rules
    bidding_rules = get_bidding_rules()
    
    # Evaluate bidding rules
    bid_adjustments = evaluate_bidding_rules(performance_data, bidding_rules)
    
    # Apply bid adjustments
    apply_bid_adjustments(bid_adjustments)
    
    # Prepare response
    response = {
        'summary': summary,
        'alerts': alerts,
        'bid_adjustments': bid_adjustments,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Completed Amazon Ads performance analysis: {len(alerts)} alerts, {len(bid_adjustments)} bid adjustments")
    
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }


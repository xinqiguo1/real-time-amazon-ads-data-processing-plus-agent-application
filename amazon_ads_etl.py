from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, from_utc_timestamp, sum as spark_sum, round as spark_round, when, lit, avg, count, expr
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import boto3
import json
from datetime import datetime, timedelta

def create_spark_session():
    """
    Create and return a Spark session configured for AWS Glue
    """
    spark = SparkSession.builder \
        .appName("Amazon Ads ETL") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .enableHiveSupport() \
        .getOrCreate()
    
    return spark

def read_data_from_s3(spark, bucket, path, format="csv"):
    """
    Read data from S3 into a Spark DataFrame
    """
    s3_path = f"s3a://{bucket}/{path}"
    
    if format == "csv":
        return spark.read.option("header", "true") \
                         .option("inferSchema", "true") \
                         .csv(s3_path)
    elif format == "parquet":
        return spark.read.parquet(s3_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def process_traffic_data(traffic_df):
    """
    Process Amazon Ads traffic data
    """
    # Convert timestamp and create date columns
    traffic_df = traffic_df.withColumn("timestamp", 
                                      from_utc_timestamp(col("time_window_start"), "UTC"))
    
    traffic_df = traffic_df.withColumn("date", date_format(col("timestamp"), "yyyy-MM-dd"))
    
    # Filter out negative impressions (data corrections)
    traffic_df = traffic_df.filter(col("impressions") >= 0)
    
    # Aggregate metrics by campaign, ad_group, and date
    traffic_agg = traffic_df.groupBy("campaign_id", "ad_group_id", "date") \
                           .agg(
                               spark_sum("impressions").alias("total_impressions"),
                               spark_sum("clicks").alias("total_clicks"),
                               spark_sum("cost").alias("total_cost"),
                               spark_round(spark_sum("cost") / spark_sum("clicks"), 2).alias("cpc")
                           )
    
    # Handle division by zero for CPC
    traffic_agg = traffic_agg.withColumn("cpc", 
                                        when(col("total_clicks") > 0, col("cpc")).otherwise(0))
    
    # Add CTR (Click-Through Rate) metric
    traffic_agg = traffic_agg.withColumn("ctr", 
                                        when(col("total_impressions") > 0,
                                             spark_round(col("total_clicks") * 100 / col("total_impressions"), 2)
                                            ).otherwise(0))
    
    return traffic_agg

def process_conversion_data(conv_df):
    """
    Process Amazon Ads conversion data
    """
    # Convert timestamp and create date columns
    conv_df = conv_df.withColumn("timestamp", 
                                from_utc_timestamp(col("time_window_start"), "UTC"))
    
    conv_df = conv_df.withColumn("date", date_format(col("timestamp"), "yyyy-MM-dd"))
    
    # Aggregate metrics by campaign, ad_group, and date
    conv_agg = conv_df.groupBy("campaign_id", "ad_group_id", "date") \
                     .agg(
                         spark_sum("attributed_conversions_1d").alias("conversions_1d"),
                         spark_sum("attributed_sales_1d").alias("sales_1d"),
                         spark_sum("attributed_units_ordered_1d").alias("units_ordered_1d")
                     )
    
    return conv_agg

def join_and_calculate_metrics(traffic_agg, conv_agg):
    """
    Join traffic and conversion data and calculate performance metrics
    """
    # Join the two datasets
    joined_df = traffic_agg.join(conv_agg, 
                                on=["campaign_id", "ad_group_id", "date"], 
                                how="left")
    
    # Fill null values for conversions
    joined_df = joined_df.fillna(0, subset=["conversions_1d", "sales_1d", "units_ordered_1d"])
    
    # Calculate ACOS (Advertising Cost of Sale)
    joined_df = joined_df.withColumn("acos", 
                                    when(col("sales_1d") > 0, 
                                         spark_round(col("total_cost") * 100 / col("sales_1d"), 2)
                                        ).otherwise(0))
    
    # Calculate ROAS (Return on Ad Spend)
    joined_df = joined_df.withColumn("roas", 
                                    when(col("total_cost") > 0, 
                                         spark_round(col("sales_1d") / col("total_cost"), 2)
                                        ).otherwise(0))
    
    # Calculate Conversion Rate
    joined_df = joined_df.withColumn("conversion_rate", 
                                    when(col("total_clicks") > 0, 
                                         spark_round(col("conversions_1d") * 100 / col("total_clicks"), 2)
                                        ).otherwise(0))
    
    return joined_df

def calculate_performance_trends(joined_df, days=7):
    """
    Calculate performance trends over time
    """
    # Create window specifications for trend calculations
    window_spec = Window.partitionBy("campaign_id", "ad_group_id") \
                        .orderBy("date") \
                        .rowsBetween(-days, 0)
    
    # Calculate moving averages for key metrics
    trend_df = joined_df.withColumn("avg_ctr_7d", avg("ctr").over(window_spec)) \
                       .withColumn("avg_conversion_rate_7d", avg("conversion_rate").over(window_spec)) \
                       .withColumn("avg_acos_7d", avg("acos").over(window_spec)) \
                       .withColumn("avg_roas_7d", avg("roas").over(window_spec))
    
    # Calculate day-over-day changes
    window_prev_day = Window.partitionBy("campaign_id", "ad_group_id") \
                            .orderBy("date") \
                            .rowsBetween(-1, -1)
    
    trend_df = trend_df.withColumn("prev_day_impressions", 
                                  F.first("total_impressions").over(window_prev_day))
    
    trend_df = trend_df.withColumn("impressions_change", 
                                  when(col("prev_day_impressions").isNotNull(),
                                       spark_round((col("total_impressions") - col("prev_day_impressions")) * 100 / 
                                        when(col("prev_day_impressions") != 0, col("prev_day_impressions")).otherwise(1), 2)
                                      ).otherwise(0))
    
    # Similar calculations for other key metrics
    trend_df = trend_df.withColumn("prev_day_cost", F.first("total_cost").over(window_prev_day))
    trend_df = trend_df.withColumn("cost_change", 
                                  when(col("prev_day_cost").isNotNull() & (col("prev_day_cost") != 0),
                                       spark_round((col("total_cost") - col("prev_day_cost")) * 100 / col("prev_day_cost"), 2)
                                      ).otherwise(0))
    
    trend_df = trend_df.withColumn("prev_day_sales", F.first("sales_1d").over(window_prev_day))
    trend_df = trend_df.withColumn("sales_change", 
                                  when(col("prev_day_sales").isNotNull() & (col("prev_day_sales") != 0),
                                       spark_round((col("sales_1d") - col("prev_day_sales")) * 100 / col("prev_day_sales"), 2)
                                      ).otherwise(0))
    
    # Drop temporary columns
    trend_df = trend_df.drop("prev_day_impressions", "prev_day_cost", "prev_day_sales")
    
    return trend_df

def identify_anomalies(trend_df):
    """
    Identify anomalies in advertising performance
    """
    # Define thresholds for anomaly detection
    ctr_threshold = 50  # 50% change in CTR
    acos_threshold = 30  # 30% change in ACOS
    conversion_threshold = 40  # 40% change in conversion rate
    
    # Detect anomalies based on day-over-day changes
    anomaly_df = trend_df.withColumn("ctr_anomaly", 
                                    when(abs(col("ctr") - F.lag("ctr", 1).over(Window.partitionBy("campaign_id", "ad_group_id").orderBy("date"))) > ctr_threshold, 1).otherwise(0))
    
    anomaly_df = anomaly_df.withColumn("acos_anomaly", 
                                      when(abs(col("acos") - F.lag("acos", 1).over(Window.partitionBy("campaign_id", "ad_group_id").orderBy("date"))) > acos_threshold, 1).otherwise(0))
    
    anomaly_df = anomaly_df.withColumn("conversion_anomaly", 
                                      when(abs(col("conversion_rate") - F.lag("conversion_rate", 1).over(Window.partitionBy("campaign_id", "ad_group_id").orderBy("date"))) > conversion_threshold, 1).otherwise(0))
    
    # Flag campaigns with anomalies
    anomaly_df = anomaly_df.withColumn("has_anomaly", 
                                      when((col("ctr_anomaly") == 1) | 
                                           (col("acos_anomaly") == 1) | 
                                           (col("conversion_anomaly") == 1), 1).otherwise(0))
    
    return anomaly_df

def write_to_s3(df, bucket, path, format="parquet", partition_cols=None):
    """
    Write DataFrame to S3
    """
    s3_path = f"s3a://{bucket}/{path}"
    
    write_options = {}
    if partition_cols:
        df.write.partitionBy(partition_cols) \
                .mode("overwrite") \
                .format(format) \
                .options(**write_options) \
                .save(s3_path)
    else:
        df.write.mode("overwrite") \
                .format(format) \
                .options(**write_options) \
                .save(s3_path)

def send_sns_alert(topic_arn, message, subject):
    """
    Send SNS alert for anomalies or threshold breaches
    """
    sns_client = boto3.client('sns')
    response = sns_client.publish(
        TopicArn=topic_arn,
        Message=message,
        Subject=subject
    )
    return response

def main():
    # Create Spark session
    spark = create_spark_session()
    
    # Define S3 paths and other configuration
    s3_bucket = "amazon-ads-data"
    traffic_path = "raw/sp_traffic"
    conversion_path = "raw/sp_conversion"
    output_path = "processed/ad_performance"
    sns_topic_arn = "arn:aws:sns:us-east-1:412381752211:Ad_Performance_Alerts"
    
    # Read data from S3
    traffic_df = read_data_from_s3(spark, s3_bucket, traffic_path)
    conv_df = read_data_from_s3(spark, s3_bucket, conversion_path)
    
    # Process data
    traffic_agg = process_traffic_data(traffic_df)
    conv_agg = process_conversion_data(conv_df)
    
    # Join and calculate metrics
    joined_df = join_and_calculate_metrics(traffic_agg, conv_agg)
    
    # Calculate trends
    trend_df = calculate_performance_trends(joined_df)
    
    # Identify anomalies
    anomaly_df = identify_anomalies(trend_df)
    
    # Write processed data to S3
    write_to_s3(anomaly_df, s3_bucket, output_path, partition_cols=["date"])
    
    # Check for anomalies and send alerts if needed
    anomalies = anomaly_df.filter(col("has_anomaly") == 1).count()
    
    if anomalies > 0:
        # Get campaigns with anomalies
        anomaly_campaigns = anomaly_df.filter(col("has_anomaly") == 1) \
                                     .select("campaign_id", "date", "ctr_anomaly", "acos_anomaly", "conversion_anomaly") \
                                     .collect()
        
        # Prepare alert message
        alert_message = f"🚨 Alert: {anomalies} campaigns have performance anomalies!\n\n"
        
        for row in anomaly_campaigns[:10]:  # Limit to first 10 anomalies
            alert_message += f"Campaign {row['campaign_id']} on {row['date']}:\n"
            if row['ctr_anomaly'] == 1:
                alert_message += "- CTR anomaly detected\n"
            if row['acos_anomaly'] == 1:
                alert_message += "- ACOS anomaly detected\n"
            if row['conversion_anomaly'] == 1:
                alert_message += "- Conversion rate anomaly detected\n"
            alert_message += "\n"
        
        # Send SNS alert
        send_sns_alert(
            sns_topic_arn,
            alert_message,
            f"Amazon Ads Performance Alert - {datetime.now().strftime('%Y-%m-%d')}"
        )
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main() 
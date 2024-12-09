# Base image
FROM openjdk:11-jre-slim

# Set the working directory
WORKDIR /app

# Install wget to download Spark
RUN apt-get update && apt-get install -y wget && apt-get clean

# Download and set up Apache Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz && \
    tar -xvf spark-3.5.3-bin-hadoop3.tgz && \
    mv spark-3.5.3-bin-hadoop3 /opt/spark && \
    rm spark-3.5.3-bin-hadoop3.tgz

# Set environment variables for Spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Copy the JAR file and models directory
COPY app.jar /app/app.jar
COPY models /app/models

# Command to run the application
CMD ["java", "-cp", "app.jar:/opt/spark/jars/*", "com.example.App"]

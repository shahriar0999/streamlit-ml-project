import pytest
from unittest import mock
import streamlit as st
import boto3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from app import download_dir  # Assuming your function is inside 'app.py'

@pytest.fixture
def mock_s3_client():
    # Mock the boto3 S3 client
    with mock.patch.object(boto3, 'client', return_value=mock.MagicMock()) as mock_client:
        yield mock_client

def test_download_model(mock_s3_client):
    # Mocking the response from S3
    mock_s3_client.return_value.get_paginator.return_value.paginate.return_value = [{
        'Contents': [{'Key': 'ml-models/tinybert-sentiment-analysis/model.bin'}]
    }]
    
    # Test the download function
    download_dir("tmp", "ml-models/tinybert-sentiment-analysis/")
    
    # Ensure the S3 client was called
    mock_s3_client.return_value.download_file.assert_called_once_with(
        'mlops-crown', 'ml-models/tinybert-sentiment-analysis/model.bin', 'tmp/model.bin'
    )

def test_text_classification():
    # Test the text classification functionality with a mock input
    text = "I love this product!"
    
    # Mock the classifier response
    with mock.patch('app.pipeline') as mock_pipeline:
        mock_pipeline.return_value = [{'label': 'POSITIVE', 'score': 0.993751585483551}]
        
        # Call the text classifier
        output = mock_pipeline("text-classification", model="tinybert-sentiment-analysis")(text)
        
        # Test the result
        assert output == [{'label': 'POSITIVE', 'score': 0.993751585483551}]

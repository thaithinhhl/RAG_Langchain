#!/bin/bash

QUESTION=$1

curl -X POST "http://localhost:5000/generative_ai" \
-H "Content-Type: application/json" \
-d "{\"question\": \"${QUESTION}\"}"

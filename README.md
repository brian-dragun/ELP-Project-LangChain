OPENAI_API_KEY: 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

https://ai-documentcompare-poc490709415051.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview
10BlITrqL1qtqbvrQqjmhwNyogM22xhLpEu0j4UKTVuADVJR3XmUJQQJ99BCACYeBjFXJ3w3AAAAACOGPMfK


AZURE_AI_SERVICES_KEY: 10BlITrqL1qtqbvrQqjmhwNyogM22xhLpEu0j4UKTVuADVJR3XmUJQQJ99BCACYeBjFXJ3w3AAAAACOGPMfK
AZURE_AI_SERVICES_ENDPOINT: https://ai-documentcompare-poc490709415051.openai.azure.com/
AZURE_AI_SERVICES_REGION: eastus


LANGCHAIN API
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

LambdaLabs API
secret_langchain-api_f5317003614d4426ad833ba9c3dbcdfe.hamV6E06g98LET6AKa2guKRpQV6TXSbJ

 http://localhost:8000/agent/playground

deactivate
rm -rf venv
python3 -m venv .venv && source .venv/bin/activate
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn AzureOpenAI_Run:app --reload

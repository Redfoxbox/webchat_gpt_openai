# webchat_gpt_openai
 webchat with custom data

docker build --tag webchat-gpt .

docker run -v $(pwd)/storage:/app/storage --env-file=./.env --publish 7860:7860 chat:latest


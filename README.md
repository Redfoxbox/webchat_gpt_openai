# webchat_gpt_openai
 webchat with custom data

docker build --tag chat .

docker run -v /home/ubuntu/chat_docker_2/storage:/app/storage --env-file=./.env --publish 7860:7860 chat:latest


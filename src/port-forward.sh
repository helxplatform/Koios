nohup kubectl -n ner port-forward svc/langfuse 3000:3000 > port-fwd.log &
nohup kubectl -n ner port-forward svc/koios-sapbert-qdrant 6333:6333 > port-fwd.log &
nohup kubectl -n ner port-forward svc/ollama 11434:11434 > port-fwd.log &
nohup kubectl -n bdc-search-dev port-forward svc/search-redis-master 6379:6379 > port-fwd.log &
nohup kubectl -n ner port-forward svc/vllm-llama-3-1-8b-instruct 9091:80 > port-fwd.log &
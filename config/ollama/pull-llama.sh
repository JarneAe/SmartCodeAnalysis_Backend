./bin/ollama serve &

pid=$!

sleep 5

echo "pulling qwen2.5:7b model"
ollama pull qwen2.5:7b
echo "pulling nomic-embed-text model"
ollama pull nomic-embed-text

echo " === ALL MODELS PULLED ==="

wait $pid

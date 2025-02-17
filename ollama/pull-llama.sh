./bin/ollama serve &

pid=$!

sleep 5

echo "pulling qwen2.5:7b model"
ollama pull qwen2.5:7b

echo " === ALL MODELS PULLED ==="

wait $pid

#!/bin/bash
set -e

NAMESPACE=ollama-chatbot

# --- Functions for managing Kubernetes resources ---
function create_namespace() {
  echo "🔧 Creating namespace: $NAMESPACE"
  kubectl create namespace $NAMESPACE || echo "Namespace already exists."
}

function apply_ollama() {
  echo "📦 Applying Ollama Deployment and Service..."
  kubectl apply -n $NAMESPACE -f k8s/ollama-pvc.yaml
  kubectl apply -n $NAMESPACE -f k8s/ollama-deployment.yaml
  kubectl apply -n $NAMESPACE -f k8s/ollama-service.yaml
}

function apply_app() {
  echo "📦 Applying App Deployment and Service..."
  kubectl apply -n $NAMESPACE -f k8s/app-deployment.yaml
  kubectl apply -n $NAMESPACE -f k8s/app-service.yaml
}

function apply_env() {
  if [ -f "./src/.env" ]; then
    echo "🔐 Creating ConfigMap from .env file..."
    kubectl create configmap app-env --from-env-file=./src/.env -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
  else
    echo "⚠️  .env file not found at ./src/.env — skipping ConfigMap creation."
  fi
}

# --- Main functions for installation and running the app ---
function k8s() {
  create_namespace
  apply_ollama
  apply_app
  apply_env
  echo "✅ Installation complete. Use 'kubectl get all -n $NAMESPACE' to check resources."
}

function setup_k8s() {
  echo "🔧 Setting up inside Kubernetes pod..."
  kubectl exec -it ollama -n $NAMESPACE -- /bin/sh -c "ollama pull llama3.1:8b && ollama pull nomic-embed-text"
  echo "✅ Setup inside Kubernetes pod complete."
}

function setup_ollama_container() {
  echo "🔧 Setting up inside Docker container..."
  docker exec -it ollama ollama pull llama3.1:8b
  docker exec -it ollama ollama pull nomic-embed-text
  echo "✅ Setup inside Docker container complete."
}

function help() {
  echo "Usage: $0 {k8s|docker|help}"
  echo "  k8s          - Create namespace and deploy all Kubernetes resources"
  echo "  setup_k8s    - Pull necessary Ollama models inside Kubernetes pod"
  echo "  docker       - Pull necessary Ollama images inside Docker container"
  echo "  help         - Show this help message"
}

case "$1" in
  k8s)
    k8s
    ;;
  setup-k8s)
    setup_k8s
    ;;
  docker)
    setup_ollama_container
    ;;
  help|*)
    help
    ;;
esac

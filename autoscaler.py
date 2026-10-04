import time
import requests
from kubernetes import client, config

PROMETHEUS_URL = "http://prometheus-kube-prometheus-prometheus.default.svc:9090/api/v1/query"
DEPLOYMENT_NAME = "ml-inference"
NAMESPACE = "default"

MIN_REPLICAS = 1
MAX_REPLICAS = 10

def query_prometheus(metric):
    response = requests.get(PROMETHEUS_URL, params={"query": metric})
    result = response.json()["data"]["result"]
    if not result:
        return 0
    return float(result[0]["value"][1])

def get_current_replicas(api):
    deploy = api.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
    return deploy.spec.replicas

def set_replicas(api, replicas):
    replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, replicas))
    body = {"spec": {"replicas": replicas}}
    api.patch_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE, body)
    print(f"Scaled to {replicas} replicas")

def autoscale_loop():
    config.load_incluster_config()  # for running inside cluster
    api = client.AppsV1Api()

    while True:
        active = query_prometheus("dispatcher_active_requests")
        current = get_current_replicas(api)

        print(f"Active requests: {active}, Replicas: {current}")

        # Scale up
        if active > current * 3:
            set_replicas(api, current + 1)

        # Scale down
        elif active < current * 1 and current > MIN_REPLICAS:
            set_replicas(api, current - 1)

        time.sleep(10)

if __name__ == "__main__":
    autoscale_loop()

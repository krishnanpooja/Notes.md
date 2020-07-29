**Docker and Kubernetes**

Docker is an open-source platform used to handle software development. Its main benefit is that it packages the settings and dependencies that the software/application needs to run into a container, which allows for portability and several other advantages. Kubernetes allows for the manual linking and orchestration of several containers, running on multiple hosts that have been created using Docker.

A fundamental difference between Kubernetes and Docker is that Kubernetes is meant to run across a cluster while Docker runs on a single node. Kubernetes is more extensive than Docker Swarm and is meant to coordinate clusters of nodes at scale in production in an efficient manner.
Kubernetes is an open-source container management tool which holds the responsibilities of container deployment, scaling & descaling of containers & load balancing.

Container- App+library

Consider you have 5-6 microservices for a single application performing various tasks, and all these microservices are put inside containers. Now, to make sure that these containers communicate with each other we need container orchestration.

Kubernetes features:
1. Auto scheduling
2. automatic healing
3. horizontal scaling and load balancing
4. automated rollouts and rollback

https://docs.docker.com/get-started/overview/

**Kubeflow**

It helps you run ML workflow on Kubernetes
Kubeflow is a platform for data scientists who want to build and experiment with ML pipelines. Kubeflow is also for ML engineers and operational teams who want to deploy ML systems to various environments for development, testing, and production-level serving.
Kubeflow is the ML toolkit for Kubernetes. 
Interface is called kfctl


**Kubernetes Cluster:**

1. Create a Cluster 
Master node and nodes which run the application
Master handles all the scheduling, deployment, scaling operations.
Nodes communicate using API exposed by the master
Minikube can be used to deploy cluster like scenario on local machines.

2. Deploy app
Kubectl - Kubernetes command line interface
When you create a Deployment, you'll need to specify the container image for your application and the number of replicas that you want to run. You can change that information later by updating your Deployment; Modules 5 and 6 of the bootcamp discuss how you can scale and update your Deployments.
kubectl create deployment command. We need to provide the deployment name and app image location (include the full repository url for images hosted outside Docker hub).
```
>kubectl create deployment kubernetes-bootcamp --image=gcr.io/google-samples/kubernetes-bootcamp:v1
>kubectl get deployment
```

3.Explore the app
Pods- A Pod is a group of one or more application containers (such as Docker or rkt) and includes shared storage (volumes), IP address and information about how to run them.
Nodes- They have multiple pods running on them. Master handles the nodes. The image is retrieved from the registry and deployed.
- kubectl get - list resources
- kubectl describe - show detailed information about a resource
- kubectl logs - print the logs from a container in a pod
- kubectl exec - execute a command on a container in a pod

4.Scale the apps
```
kubectl scale deployments/kubernetes-bootcamp --replicas=4
```

5. Update the apps
Rolling updates allow Deployments' update to take place with zero downtime by incrementally updating Pods instances with new ones.

# StyleForge
## Introduction
Introducing our service that utilizes the power of neural networks to generate unique and personalized clothing styles for you. Our service takes into account your preferences to create styles that perfectly suit your individual taste and needs. With the help of our advanced algorithms, we can search through a vast selection of online shops and recommend clothing items that match your generated style, saving you time and effort in the shopping process. Our service combines the best of technology and fashion to provide you with a seamless and enjoyable shopping experience. Try it out today and revolutionize the way you shop for clothes!

## Description
Our service consists of 2 parts: generation engine and search engine.

### Generation engine
In development.
### Search engine
For a search engine we solve two task: detection of a clothing item on image and metric learning task, on a database of images from DeepFashion 2 dataset. 
In fututre we plan to implement parsing online stores and showing users links to existing webpages with clothes.

## Datasets
### Generation
1. [Fashion-MMT](https://github.com/syuqings/Fashion-MMT)
2. [FashionGen](https://arxiv.org/pdf/1806.08317v1.pdf)
### Detection
1. [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
2. [DeepFashion2](https://github.com/switchablenorms/DeepFashion2)
### Metric Learning 
1. [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
2. [DeepFashion2](https://github.com/switchablenorms/DeepFashion2)

## Architecture draft
![xd](resources/StyleForge.drawio.png)


## API limits

* Input image file size 5 MB maximum
* The minimum image dimensions is 80 pixels for both height and width
* The maximum image dimensions is 1024 pixels for both height and width
* JSON request object size 10 MB (in case of generation)
* Images per request 1
* Service supports the PNG and JPEG image formats
* Response limit - 10 images

Time of responce
* Responses from the API can take anywhere from <10 second to 30 seconds


## Metrics
* Recall@k
* Precision@k
* F1@k

## Project scalability
* We provide load balancer between frontend and backend layers
* Our search and generation services are packed in docker containers
* We provide Kubernetes cluster which manages the search and generation containers
* Embedding space storage is hosted as PostgreSQL, so we can use some built-in solution for load balancing and parallel query execution.

## Demo
You can try running our API in docker. NVIDIA GPU environment with CUDA is required.
### Step 1
[Download](https://drive.google.com/drive/folders/108w6526DIhWNO5WDkgMcGwFPvjGFdf7N?usp=sharing) models, embeddings database and dataframe with metainfo.
### Step 2
In ```/clean_app/config.py``` put pathes to model, embedding and metainfo dataframe.
### Step 3
Build and run docker container.
```commandline
cd <path_to_cloned_repo>/clear_app
docker build -t styleforge . 
docker run --name styleforge_test --gpus all --publish 80:80 styleforge
```

> **Note:**
> Tested on Windows environment only.

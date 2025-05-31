# Introduction

This project is by Ayoub CHOUKRI, Axel OLOGOUNA, Qian WANG, and Minh-Duy NGUYEN.

This repository contains the source code for our web application related to the AI Frameworks project. It is designed to be used with Docker and Docker Compose, facilitating the deployment and management of the application.

# Project Organization

The project contains several directories and files organized as follows:

-   **Docker**: Contains all the necessary files to build the Docker image for the application:
    -   A **Dockerfiles** folder: Contains two Dockerfiles, `Dockerfile.webapp` and `Dockerfile.model`, to build the images for the web application and the AI model, respectively.
    -   A **DockerCompose** folder: Contains the `docker-compose.yml` file that defines the services, networks, and volumes needed to run the application.
    -   A **Requirements** folder: Contains `requirements_webapp.txt` and `requirements_model.txt` files, which list the Python dependencies required for the web application and the AI model, respectively.

-   **Webapp**: Contains the source code for the web application, including HTML, CSS, JavaScript, and Python files necessary to run the application.

-   **Modules**: Contains Python modules used by the application and utilities necessary for data processing and AI model operation.

-   **Models**: Contains Python scripts to import, build AI models, and use them in the application.

-   **Saved_Models**: Contains pre-trained and saved AI models, ready to be used by the application.

-   **Main_Scripts**: Contains the main scripts for training models, building Annoy bases, etc.

-   **Saved_Metrics**: Contains saved metrics to evaluate the performance of AI models.


# Running the Application

## Building Docker Images

To build the Docker images for the application, you have two options:

* Execute the `Build_Images.sh` script. This script will build the Docker images for the web application and the AI model using the provided Dockerfiles. It must be run from the project's root directory.
    To execute it, use the `source` command in your terminal (from the project root):
    ```bash
    source Build_Images.sh
    ```
    (This command executes the script in the current shell. Alternatively, you could run it in a subshell using `bash Build_Images.sh` or, after making it executable with `chmod +x Build_Images.sh`, using `./Build_Images.sh`.)

* Or:
    -   Navigate to the `Docker/DockerCompose` directory and run the following command:
        ```bash
        docker compose build
        ```


## Launching Containers

To launch the containers, you have two options:

* Execute the `Launch_Docker_Containers.sh` script. This script will launch the containers for the web application and the AI model using the provided `docker-compose.yml` file. It must be run from the project's root directory.
    To execute it, use the `source` command in your terminal (from the project root):
    ```bash
    source Launch_Docker_Containers.sh
    ```
    (This command executes the script in the current shell. Alternatively, you could run it in a subshell using `bash Launch_Docker_Containers.sh` or, after making it executable with `chmod +x Launch_Docker_Containers.sh`, using `./Launch_Docker_Containers.sh`.)

* Or:
    -   Navigate to the `Docker/DockerCompose` directory and run the following command:
        ```bash
        docker compose up
        ```
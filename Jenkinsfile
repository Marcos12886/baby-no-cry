pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "docker-aplicacion"
        DOCKER_REGISTRY = "ghcr.io/marcos12886"
        GIT_CREDENTIALS_ID = "GIT_CREDENTIALS_ID"
    }

    stages {

        stage('Clonar repositorio') {
            steps {
                git credentialsId: "${GIT_CREDENTIALS_ID}",  url: 'https://github.com/Marcos12886/baby-no-cry'
            }
        }
        stage('Crear Imagen de Docker') {
            steps {
                sh 'docker build -t docker-aplicacion .'
            }
        }
        stage('Pushear imagen de Docker al GitHub Registry') {
            steps {
                script {
                    docker.withRegistry('https://ghcr.io', 'GIT_CREDENTIALS_ID') {
                        sh 'docker tag ${DOCKER_IMAGE}:latest ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest'
                        sh 'docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest'
                    }
                }
            }
        }
        stage('Run application') {
            steps {
                sh 'docker run -d -p 7860:7860 --name contenedor-app docker-aplicacion'
            }
        }
    }
}

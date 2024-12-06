pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "docker-aplicacion"
        GIT_CREDENTIALS_ID = "GIT_CREDENTIALS_ID"
    }

    stages {
        stage('Crear Imagen de Docker') {
            steps {
                sh 'docker build -t docker-aplicacion .'
                }
            }
        stage('Run application') {
            steps {
                sh 'docker run -d -p 7860:7860 --name contenedor-app docker-aplicacion'
                }
            }
        }
    }

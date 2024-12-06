pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "docker-aplicacion"
        GIT_CREDENTIALS_ID = "GIT_CREDENTIALS_ID"
    }

    stages {

        stage('Clonar repositorio') {
            steps {
                git credentialsId: "${GIT_CREDENTIALS_ID}", url: 'https://github.com/Marcos12886/baby-no-cry'
            }
        }

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

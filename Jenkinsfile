pipeline {
    agent any
    stages {
        stage('version') {
            steps {
                sh 'python3.10 --version'
            }
        }
        stage('Crear Imagen de Docker') {
            steps {
                sh 'sudo docker build -t docker-aplicacion .'
                }
            }
        stage('Run application') {
            steps {
                sh 'sudo docker run -d -p 7860:7860 --name contenedor-app docker-aplicacion'
                }
            }
        }
    }

pipeline {
    agent any
    stages {
        stage('Crear Imagen de Docker') {
            steps {
                sh 'ls'
                sh 'cat src/pages/Home.jsx'
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

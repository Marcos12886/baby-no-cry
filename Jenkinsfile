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
                script {
                    docker.build('your-app-name')
                }
            }
        }
        stage('Run application') {
            steps {
                script {
                    docker.image('your-app-name').run('-p 7860:7860')
                }
            }
        }
    }
}

pipeline {
    agent any
    stages {
        stage('version') {
            steps {
                sh 'python3 --version'
            }
        }
        stage('Crear entorno') {
            steps {
                sh 'python3 -m venv venv'
            }
        }
        stage('Activar entorno') {
            steps {
                sh '. venv/bin/activate'
            }
        }
        stage('Install Dependencies') {
            steps {
                sh '/usr/bin/python3 -m pip install --user -r requirements.txt'
            }
        }
        stage('hello') {
            steps {
                sh 'python3 archivo.py'
            }
        }
    }
}

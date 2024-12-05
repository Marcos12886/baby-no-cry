pipeline {
    agent any
    stages {
        stage('version') {
            steps {
                sh 'python3.10 --version'
            }
        }
        stage('Crear entorno') {
            steps {
                sh '''
                    python3.10 -m venv entorno
                    . entorno/bin/activate
                    pip install -r requirements.txt
                    python3.10 archivo.py
                    '''
            }
        }
    }
}

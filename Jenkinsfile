pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "docker-aplicacion"
        DOCKER_REGISTRY = "ghcr.io/marcos12886"
        GIT_CREDENTIALS_ID = "GIT_CREDENTIALS_ID"
        SSH_CREDENTIALS_ID = "key"
        SERVER_IP = "172.20.0.3"
        SERVER_USER = "root"
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
        stage('SSH al servidor Ubuntu en local y Deploy') {
            steps {
                script {
                    sshagent([SSH_CREDENTIALS_ID]) {
                        sh """
                        ssh -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_IP} << EOF
                            docker login ghcr.io -u marcos12886 -p ${GIT_CREDENTIALS_ID}
                            docker stop ${DOCKER_IMAGE} || true
                            docker rm ${DOCKER_IMAGE} || true
                            docker rmi ${DOCKER_IMAGE} || true
                            docker rmi ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest || true
                            docker pull ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest
                            docker run -d --name ${DOCKER_IMAGE} -p 7861:7861 ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest
                        """
                    }
                }
            }
        }
    }
    post {
        always {
            cleanWs()
        }
    }
}

pipeline {
	agent any

	environment {
		DOCKER_IMAGE_NAME = 'ima_bebe'
		DOCKER_CONTAINER_NAME = 'cont_bebe'
	}
	stages {
		stage('Build docker image') {
			steps {
				script {
					sh "docker build -t ${DOCKER_IMAGE_NAME} ."
				}
			}
		}
		stage('Run Docker Container') {
			steps {
				script {
					sh """
						docker stop ${DOCKER_CONTAINER_NAME} || true
						docker rm ${DOCKER_CONTAINER_NAME} || true
						docker run -d -p 7860:7860 --name ${DOCKER_CONTAINER_NAME} ${DOCKER_IMAGE_NAME}
					"""

				}
			}
		}
	}
	post {
		always {
			// borrar el contenedor
			sh "docker rm -f ${DOCKER_CONTAINER_NAME} || true"
		}
	}
}

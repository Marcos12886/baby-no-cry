pipeline {
	agent any

	environment {
		DOCKER_IMAGE_NAME = 'ima_bebe'
		DOCKER_CONTAINER_NAME = 'cont_bebe'
	}
	stages {
		stage('Build docker image') {
			steps {
				sh "docker build -t ${DOCKER_IMAGE_NAME} ."
				}
			}
		}
		stage('Run Docker Container') {
			steps {
				sh "docker run -d -p 7860:7860 --name ${DOCKER_CONTAINER_NAME} ${DOCKER_IMAGE_NAME}"
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

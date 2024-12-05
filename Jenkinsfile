pipeline {
    agent any

    stages {
        stage('version') {
            steps {
                // Check Python version
                sh 'python --version'
            }
        }
        stage('Install Dependencies') {
            steps {
                // Assuming you have a requirements.txt file for your Python app
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Run Application') {
            steps {
                // Run your Python app in the background. 
                // We use nohup to prevent the script from hanging if the console is closed, 
                // and '&' to run it in the background.
                sh 'nohup python app.py &'

                // Optional: Give it a second to start up before checking if it's running
                sh 'sleep 5'

                // Check if the app is running by looking for a process ID
                sh 'ps aux | grep "[p]ython app.py" || echo "App did not start or has crashed"'
            }
        }
    }
}

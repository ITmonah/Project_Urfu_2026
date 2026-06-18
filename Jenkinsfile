pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        IMAGE_NAME = 'project-urfu-2026'
        PYTHON_BIN = 'python3.11'
        DVC_SITE_CACHE_DIR = "${WORKSPACE}/.dvc/site-cache"
        DVC_NO_ANALYTICS = '1'
    }

    stages {
        stage('Install CI dependencies') {
            steps {
                sh '''
                    set -eu
                    ${PYTHON_BIN} -m venv .venv || python3 -m venv .venv
                    . .venv/bin/activate
                    python -m pip install --upgrade pip
                    pip install -r requirements-ci.txt
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '''
                    set -eu
                    . .venv/bin/activate
                    ruff check \
                        fastapi_app \
                        pipeline/classifier_models.py \
                        pipeline/pipeline.py \
                        predict_image.py \
                        NewClassificatorNoDino/pipeline.py \
                        SAM_model/pipeline.py \
                        SMP_model/pipeline.py \
                        tests
                '''
            }
        }

        stage('Unit tests') {
            steps {
                sh '''
                    set -eu
                    . .venv/bin/activate
                    pytest -q -m "not data_quality"
                '''
            }
        }

        stage('DVC pull') {
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'dvc-s3',
                        usernameVariable: 'AWS_ACCESS_KEY_ID',
                        passwordVariable: 'AWS_SECRET_ACCESS_KEY'
                    )
                ]) {
                    sh '''
                        set -eu
                        . .venv/bin/activate
                        if [ -n "${DVC_S3_ENDPOINT_URL:-}" ]; then
                            dvc remote modify --local storage endpointurl "$DVC_S3_ENDPOINT_URL"
                        fi
                        dvc pull datasets.dvc model_weights.dvc
                        dvc status --cloud-remote storage
                    '''
                }
            }
        }

        stage('Data quality tests') {
            steps {
                sh '''
                    set -eu
                    . .venv/bin/activate
                    pytest -q -m data_quality
                '''
            }
        }

        stage('Docker build') {
            steps {
                sh '''
                    set -eu
                    docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} .
                '''
            }
        }

        stage('Docker push') {
            when {
                anyOf {
                    branch 'main'
                    buildingTag()
                }
            }
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'dockerhub',
                        usernameVariable: 'DOCKERHUB_USERNAME',
                        passwordVariable: 'DOCKERHUB_TOKEN'
                    )
                ]) {
                    sh '''
                        set -eu
                        IMAGE_REF="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${BUILD_NUMBER}"
                        docker login -u "$DOCKERHUB_USERNAME" -p "$DOCKERHUB_TOKEN"
                        docker tag "${IMAGE_NAME}:${BUILD_NUMBER}" "$IMAGE_REF"
                        docker push "$IMAGE_REF"
                        if [ "${BRANCH_NAME:-}" = "main" ]; then
                            docker tag "${IMAGE_NAME}:${BUILD_NUMBER}" "${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest"
                            docker push "${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest"
                        fi
                    '''
                }
            }
        }
    }
}

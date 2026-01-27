$ErrorActionPreference = "Stop"

# Configuration
$PROJECT_ID = "fretadonow-saas"
$REGION = "us-central1"
$SERVICE_NAME = "neurotriage-ai"
$SERVICE_ACCOUNT = "neurotriage-sa"
$SA_EMAIL = "$SERVICE_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com"

Write-Host "Starting manual deployment for $SERVICE_NAME..."

# 1. Enable APIs (Just in case)
Write-Host "Enabling APIs..."
gcloud services enable pubsub.googleapis.com run.googleapis.com secretmanager.googleapis.com aiplatform.googleapis.com artifactregistry.googleapis.com --project $PROJECT_ID

# 2. Create Service Account
Write-Host "Creating Service Account..."
if (!(gcloud iam service-accounts list --project $PROJECT_ID --filter="email:$SA_EMAIL" --format="value(email)")) {
    gcloud iam service-accounts create $SERVICE_ACCOUNT --display-name "NeuroTriage AI Service Account" --project $PROJECT_ID
} else {
    Write-Host "Service Account $SA_EMAIL already exists."
}

# 3. Grant IAM Roles to Service Account
Write-Host "Granting IAM Roles..."
$ROLES = @(
    "roles/pubsub.subscriber",
    "roles/pubsub.publisher",
    "roles/aiplatform.user",
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
)

foreach ($role in $ROLES) {
    gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_EMAIL" --role="$role" --condition=None --quiet > $null
}

# 4. Create Pub/Sub Topics and Subscriptions
Write-Host "Creating Pub/Sub Resources..."
$TOPICS = @("audio-input", "results", "emergency-alerts", "dead-letter")

foreach ($topicSuffix in $TOPICS) {
    $topicName = "$SERVICE_NAME-$topicSuffix"
    if (!(gcloud pubsub topics list --project $PROJECT_ID --filter="name:projects/$PROJECT_ID/topics/$topicName" --format="value(name)")) {
        gcloud pubsub topics create $topicName --project $PROJECT_ID
    }
}

# Subscription for audio-input
$SUB_NAME = "$SERVICE_NAME-audio-sub"
$TOPIC_NAME = "$SERVICE_NAME-audio-input"
if (!(gcloud pubsub subscriptions list --project $PROJECT_ID --filter="name:projects/$PROJECT_ID/subscriptions/$SUB_NAME" --format="value(name)")) {
    gcloud pubsub subscriptions create $SUB_NAME --topic=$TOPIC_NAME --project=$PROJECT_ID --ack-deadline=60 --message-retention-duration=7d
}

# 5. Create Secrets
Write-Host "Creating Secrets..."
$SECRET_ID = "$SERVICE_NAME-pii-salt"
if (!(gcloud secrets list --project $PROJECT_ID --filter="name:$SECRET_ID" --format="value(name)")) {
    gcloud secrets create $SECRET_ID --replication-policy="automatic" --project $PROJECT_ID
    # Add a dummy version if empty? No, code should handle it or we set it manually.
    # echo "somesaltvalue" | gcloud secrets versions add $SECRET_ID --data-file=-
}

# 6. Deploy Backend (Processor)
Write-Host "Deploying Backend Processor to Cloud Run..."
# Note: Using previously built image from Cloud Build
$IMAGE_BACKEND = "$REGION-docker.pkg.dev/$PROJECT_ID/$SERVICE_NAME/processor:latest"
gcloud run deploy "$SERVICE_NAME-processor" `
    --image=$IMAGE_BACKEND `
    --region=$REGION `
    --project=$PROJECT_ID `
    --service-account=$SA_EMAIL `
    --allow-unauthenticated `
    --set-env-vars=GCP_PROJECT_ID=$PROJECT_ID,GCP_REGION=$REGION,ENVIRONMENT=prod,PUBSUB_SUBSCRIPTION=$SUB_NAME,DEEPGRAM_API_KEY=${env:TF_VAR_deepgram_api_key},PINECONE_API_KEY=${env:TF_VAR_pinecone_api_key},MEDGEMMA_ENABLED=true `
    --timeout=300

# 7. Deploy Frontend
Write-Host "Deploying Frontend to Cloud Run..."
$IMAGE_FRONTEND = "$REGION-docker.pkg.dev/$PROJECT_ID/$SERVICE_NAME/frontend:latest"
gcloud run deploy "$SERVICE_NAME-frontend" `
    --image=$IMAGE_FRONTEND `
    --region=$REGION `
    --project=$PROJECT_ID `
    --service-account=$SA_EMAIL `
    --allow-unauthenticated `
    --set-env-vars=PORT=8080 `
    --execution-environment=gen2

Write-Host "Deployment Complete!"
